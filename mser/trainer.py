import json
import os
import platform
import shutil
import time
from datetime import timedelta

import joblib
import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from tqdm import tqdm
from visualdl import LogWriter

from mser import SUPPORT_MODEL, __version__
from mser.data_utils.collate_fn import collate_fn
from mser.data_utils.featurizer import AudioFeaturizer
from mser.data_utils.reader import CustomDataset
from mser.metric.metrics import accuracy
from mser.models.base_model import BaseModel
from mser.models.bi_lstm import BiLSTM
from mser.utils.logger import setup_logger
from mser.utils.scheduler import WarmupCosineSchedulerLR
from mser.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments

logger = setup_logger(__name__)


class MSERTrainer(object):
    def __init__(self, configs, use_gpu=True):
        """ mser集成工具类

        :param configs: 配置字典
        :param use_gpu: 是否使用GPU训练模型
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.model = None
        self.test_loader = None
        self.amp_scaler = None
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动使用单线程读取！')
        if self.configs.preprocess_conf.feature_method == 'Emotion2Vec':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Emotion2Vec特征提取方法不支持多线程，已自动使用单线程提取特征！')
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]

    def __setup_dataloader(self, is_train=False):
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        if is_train:
            self.train_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.train_list,
                                               audio_featurizer=self.audio_featurizer,
                                               scaler_path=self.configs.dataset_conf.scaler_path,
                                               do_vad=self.configs.dataset_conf.do_vad,
                                               max_duration=self.configs.dataset_conf.max_duration,
                                               min_duration=self.configs.dataset_conf.min_duration,
                                               aug_conf=self.configs.dataset_conf.aug_conf,
                                               sample_rate=self.configs.dataset_conf.sample_rate,
                                               use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                               target_dB=self.configs.dataset_conf.target_dB,
                                               mode='train')
            # 设置支持多卡训练
            train_sampler = None
            if torch.cuda.device_count() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           **self.configs.dataset_conf.dataLoader)
        # 获取测试数据
        self.test_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.test_list,
                                          audio_featurizer=self.audio_featurizer,
                                          scaler_path=self.configs.dataset_conf.scaler_path,
                                          do_vad=self.configs.dataset_conf.do_vad,
                                          max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                          min_duration=self.configs.dataset_conf.min_duration,
                                          sample_rate=self.configs.dataset_conf.sample_rate,
                                          use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                          target_dB=self.configs.dataset_conf.target_dB,
                                          mode='eval')
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      collate_fn=collate_fn,
                                      batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                      num_workers=self.configs.dataset_conf.dataLoader.num_workers)

    # 获取用于归一化的文件
    def get_standard_file(self):
        audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                           method_args=self.configs.preprocess_conf.get('method_args', {}))
        # 获取测试数据
        test_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.train_list,
                                     audio_featurizer=audio_featurizer,
                                     do_vad=self.configs.dataset_conf.do_vad,
                                     max_duration=self.configs.dataset_conf.max_duration,
                                     min_duration=self.configs.dataset_conf.min_duration,
                                     sample_rate=self.configs.dataset_conf.sample_rate,
                                     use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                     target_dB=self.configs.dataset_conf.target_dB,
                                     mode='create_data')
        data = []
        for i in tqdm(range(len(test_dataset))):
            feature, _ = test_dataset[i]
            data.append(feature)
        scaler = StandardScaler().fit(data)
        joblib.dump(scaler, self.configs.dataset_conf.scaler_path)
        logger.info(f'归一化文件保存在：{self.configs.dataset_conf.scaler_path}')

    # 提取特征保存文件
    def extract_features(self, save_dir='dataset/features'):
        os.makedirs(save_dir, exist_ok=True)
        audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                           method_args=self.configs.preprocess_conf.get('method_args', {}))
        for i, data_list in enumerate([self.configs.dataset_conf.train_list, self.configs.dataset_conf.test_list]):
            max_duration = self.configs.dataset_conf.max_duration if i == 0 else self.configs.dataset_conf.eval_conf.max_duration
            # 获取测试数据
            test_dataset = CustomDataset(data_list_path=data_list,
                                         audio_featurizer=audio_featurizer,
                                         do_vad=self.configs.dataset_conf.do_vad,
                                         max_duration=max_duration,
                                         min_duration=self.configs.dataset_conf.min_duration,
                                         sample_rate=self.configs.dataset_conf.sample_rate,
                                         use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                         target_dB=self.configs.dataset_conf.target_dB,
                                         mode='create_data')
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(test_dataset))):
                    feature, label = test_dataset[i]
                    label = int(label)
                    save_path = os.path.join(save_dir, f'{int(time.time() * 1000)}.npy')
                    np.save(save_path, feature)
                    f.write(f'{save_path}\t{label}\n')
            logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')

    # 获取模型
    def __setup_model(self, input_size, is_train=False):
        # 自动获取列表数量
        if self.configs.model_conf.num_class is None:
            self.configs.model_conf.num_class = len(self.class_labels)
        # 获取模型
        if self.configs.use_model == 'BiLSTM':
            self.model = BiLSTM(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'BaseModel':
            self.model = BaseModel(input_size=input_size, **self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        self.model.to(self.device)
        summary(self.model, input_size=(1, self.test_dataset.audio_featurizer.feature_dim))
        # 使用Pytorch2.0的编译器
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() == 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")
        # print(self.model)
        # 获取损失函数
        weight = torch.tensor(self.configs.train_conf.loss_weight, dtype=torch.float, device=self.device) \
            if self.configs.train_conf.loss_weight is not None else None
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            # 获取优化方法
            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=self.configs.optimizer_conf.learning_rate,
                                                  weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=self.configs.optimizer_conf.learning_rate,
                                                   weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.configs.optimizer_conf.get('momentum', 0.9),
                                                 lr=self.configs.optimizer_conf.learning_rate,
                                                 weight_decay=self.configs.optimizer_conf.weight_decay)
            else:
                raise Exception(f'不支持优化方法：{optimizer}')
            # 学习率衰减函数
            scheduler_args = self.configs.optimizer_conf.get('scheduler_args', {}) \
                if self.configs.optimizer_conf.get('scheduler_args', {}) is not None else {}
            if self.configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
                max_step = int(self.configs.train_conf.max_epoch * 1.2) * len(self.train_loader)
                self.scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                                   T_max=max_step,
                                                   **scheduler_args)
            elif self.configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR':
                self.scheduler = WarmupCosineSchedulerLR(optimizer=self.optimizer,
                                                         fix_epoch=self.configs.train_conf.max_epoch,
                                                         step_per_epoch=len(self.train_loader),
                                                         **scheduler_args)
            else:
                raise Exception(f'不支持学习率衰减函数：{self.configs.optimizer_conf.scheduler}')

    def __load_pretrained(self, pretrained_model):
        # 加载预训练模型
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pth')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_dict = self.model.module.state_dict()
            else:
                model_dict = self.model.state_dict()
            model_state_dict = torch.load(pretrained_model)
            # 过滤不存在的参数
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_acc = 0
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pth')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pth')), "优化方法参数文件不存在！"
            state_dict = torch.load(os.path.join(resume_model, 'model.pth'))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pth')))
            # 自动混合精度参数
            if self.amp_scaler is not None and os.path.exists(os.path.join(resume_model, 'scaler.pth')):
                self.amp_scaler.load_state_dict(torch.load(os.path.join(resume_model, 'scaler.pth')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_acc = json_data['accuracy']
            self.optimizer.step()
            [self.scheduler.step() for _ in range(last_epoch * len(self.train_loader))]
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        return last_epoch, best_acc

    # 保存模型
    def __save_checkpoint(self, save_model_path, epoch_id, best_acc=0., best_model=False):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(state_dict, os.path.join(model_path, 'model.pth'))
        # 自动混合精度参数
        if self.amp_scaler is not None:
            torch.save(self.amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_epoch": epoch_id, "accuracy": best_acc, "version": __version__}
            f.write(json.dumps(data))
        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, epoch_id, local_rank, writer, nranks=0):
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        sum_batch = len(self.train_loader) * self.configs.train_conf.max_epoch
        for batch_id, (features, label, input_lens_ratio) in enumerate(self.train_loader):
            if nranks > 1:
                features = features.to(local_rank)
                label = label.to(local_rank).long()
            else:
                features = features.to(self.device)
                label = label.to(self.device).long()
            # 执行模型计算，是否开启自动混合精度
            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):
                output = self.model(features)
            # 计算损失值
            los = self.loss(output, label)
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(los)
                scaled.backward()
            else:
                los.backward()
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                self.amp_scaler.unscale_(self.optimizer)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            # 计算准确率
            acc = accuracy(output, label)
            accuracies.append(acc)
            loss_sum.append(los.data.cpu().numpy())
            train_times.append((time.time() - start) * 1000)

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                        sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                            f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), self.train_step)
                writer.add_scalar('Train/Accuracy', (sum(accuracies) / len(accuracies)), self.train_step)
                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_step)
                train_times, accuracies, loss_sum = [], [], []
                self.train_step += 1
            start = time.time()
            self.scheduler.step()

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        # 获取有多少张显卡训练
        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # 初始化NCCL环境
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])

        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.test_dataset.audio_featurizer.feature_dim, is_train=True)

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.__load_pretrained(pretrained_model=pretrained_model)
        # 加载恢复模型
        last_epoch, best_acc = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step, self.train_step = 0, 0
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0:
                logger.info('=' * 70)
                loss, acc = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), loss, acc))
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', acc, test_step)
                writer.add_scalar('Test/Loss', loss, test_step)
                test_step += 1
                self.model.train()
                # # 保存最优模型
                if acc >= best_acc:
                    best_acc = acc
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=acc,
                                           best_model=True)
                # 保存模型
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=acc)

    def evaluate(self, resume_model=None, save_matrix_path=None):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_matrix_path: 保存混合矩阵的路径
        :return: 评估结果
        """
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.test_dataset.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model

        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (features, label, input_lens_ratio) in enumerate(tqdm(self.test_loader)):
                features = features.to(self.device)
                label = label.to(self.device).long()
                output = eval_model(features)
                los = self.loss(output, label)
                # 计算准确率
                acc = accuracy(output, label)
                accuracies.append(acc)
                # 模型预测标签
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())
                # 真实标签
                labels.extend(label.tolist())
                losses.append(los.data.cpu().numpy())
        loss = float(sum(losses) / len(losses))
        acc = float(sum(accuracies) / len(accuracies))
        # 保存混合矩阵
        if save_matrix_path is not None:
            cm = confusion_matrix(labels, preds)
            plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                  class_labels=self.class_labels)

        self.model.train()
        return loss, acc

    def export(self, save_model_path='models/', resume_model='models/BiLSTM_Emotion2Vec/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        self.__setup_model(input_size=self.test_dataset.audio_featurizer.feature_dim)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.export()
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pth')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))
