import os
import sys
from io import BufferedReader
from typing import List

import joblib
import numpy as np
import torch
import yaml

from loguru import logger
from yeaudio.audio import AudioSegment
from mser import SUPPORT_EMOTION2VEC_MODEL
from mser.data_utils.featurizer import AudioFeaturizer
from mser.models import build_model
from mser.utils.utils import dict_to_object, print_arguments, convert_string_based_on_type


class MSERPredictor:
    def __init__(self,
                 configs,
                 use_ms_model=None,
                 model_path='models/BiLSTM_Emotion2Vec/best_model/',
                 use_gpu=True,
                 overwrites=None,
                 log_level="info"):
        """语音情感训练工具类

        :param configs: 配置文件路径，或者模型名称，如果是模型名称则会使用默认的配置文件
        :param use_ms_model: 使用ModelScope上公开Emotion2vec的模型
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        :param overwrites: 覆盖配置文件中的参数，比如"train_conf.max_epoch=100"，多个用逗号隔开
        :param log_level: 打印的日志等级，可选值有："debug", "info", "warning", "error"
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.log_level = log_level.upper()
        logger.remove()
        logger.add(sink=sys.stdout, level=self.log_level)
        self.use_ms_model = use_ms_model
        # 使用ModelScope上的模型
        if use_ms_model is not None:
            # 支持的模型
            assert use_ms_model in SUPPORT_EMOTION2VEC_MODEL, f'没有该模型：{use_ms_model}'
            from mser.utils.emotion2vec_predict import Emotion2vecPredict
            self.predictor = Emotion2vecPredict(use_ms_model, revision=None, use_gpu=use_gpu)
            return
        # 读取配置文件
        if isinstance(configs, str):
            # 获取当前程序绝对路径
            absolute_path = os.path.dirname(__file__)
            # 获取默认配置文件路径
            config_path = os.path.join(absolute_path, f"configs/{configs}.yml")
            configs = config_path if os.path.exists(config_path) else configs
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.configs = dict_to_object(configs)
        # 覆盖配置文件中的参数
        if overwrites:
            overwrites = overwrites.split(",")
            for overwrite in overwrites:
                keys, value = overwrite.strip().split("=")
                attrs = keys.split('.')
                current_level = self.configs
                for attr in attrs[:-1]:
                    current_level = getattr(current_level, attr)
                before_value = getattr(current_level, attrs[-1])
                setattr(current_level, attrs[-1], convert_string_based_on_type(before_value, value))
        # 打印配置信息
        print_arguments(configs=self.configs)
        # 获取特征器
        self._audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                 method_args=self.configs.preprocess_conf.get('method_args', {}))
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        # 自动获取列表数量
        if self.configs.model_conf.model_args.get('num_class', None) is None:
            self.configs.model_conf.model_args.num_class = len(self.class_labels)
        # 获取模型
        self.predictor = build_model(input_size=self._audio_featurizer.feature_dim, configs=self.configs)
        self.predictor.to(self.device)
        # 加载模型
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pth')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        if torch.cuda.is_available() and use_gpu:
            model_state_dict = torch.load(model_path, weights_only=False)
        else:
            model_state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.predictor.load_state_dict(model_state_dict)
        print(f"成功加载模型参数：{model_path}")
        self.predictor.eval()
        # 加载归一化文件
        self.scaler = joblib.load(self.configs.dataset_conf.dataset.scaler_path)

    def _load_audio(self, audio_data, sample_rate=16000):
        """加载音频
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        # 重采样
        if audio_segment.sample_rate != self.configs.dataset_conf.dataset.sample_rate:
            audio_segment.resample(self.configs.dataset_conf.dataset.sample_rate)
        # decibel normalization
        if self.configs.dataset_conf.dataset.use_dB_normalization:
            audio_segment.normalize(target_db=self.configs.dataset_conf.dataset.target_dB)
        assert audio_segment.duration >= self.configs.dataset_conf.dataset.min_duration, \
            f'音频太短，最小应该为{self.configs.dataset_conf.dataset.min_duration}s，当前音频为{audio_segment.duration}s'
        # 获取特征
        feature = self._audio_featurizer(audio_segment.samples, sample_rate=audio_segment.sample_rate)
        # 归一化
        feature = self.scaler.transform([feature])
        feature = feature.squeeze().astype(np.float32)
        return feature

    # 预测一个音频的特征
    def predict(self,
                audio_data,
                sample_rate=16000):
        """预测一个音频

        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 结果标签和对应的得分
        """
        if self.use_ms_model is not None:
            labels, scores = self.predictor.predict(audio_data)
            return labels[0], scores[0]
        # 加载音频文件，并进行预处理
        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device).unsqueeze(0)
        # 执行预测
        output = self.predictor(input_data)
        result = torch.nn.functional.softmax(output, dim=-1)[0]
        result = result.data.cpu().numpy()
        # 最大概率的label
        lab = np.argsort(result)[-1]
        score = result[lab]
        return self.class_labels[lab], round(float(score), 5)

    def predict_batch(self, audios_data: List, sample_rate=16000):
        """预测一批音频的特征

        :param audios_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 结果标签和对应的得分
        """
        if self.use_ms_model is not None:
            labels, scores = self.predictor.predict(audios_data)
            return labels, scores
        audios_data1 = []
        for audio_data in audios_data:
            # 加载音频文件，并进行预处理
            input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
            audios_data1.append(input_data)
        # 找出音频长度最长的
        batch = sorted(audios_data1, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
        for x in range(batch_size):
            tensor = audios_data1[x]
            seq_length = tensor.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[x, :seq_length] = tensor[:]
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        # 执行预测
        output = self.predictor(inputs)
        results = torch.nn.functional.softmax(output, dim=-1)
        results = results.data.cpu().numpy()
        labels, scores = [], []
        for result in results:
            lab = np.argsort(result)[-1]
            score = result[lab]
            labels.append(self.class_labels[lab])
            scores.append(round(float(score), 5))
        return labels, scores
