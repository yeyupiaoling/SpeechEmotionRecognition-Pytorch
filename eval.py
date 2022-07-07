import argparse
import functools

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data_utils.reader import CustomDataset
from modules.model import Model
from utils.utility import add_arguments, print_arguments, plot_confusion_matrix

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_class',        int,    6,                        '分类的类别数量')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('label_list_path',   str,   'dataset/label_list.txt', '标签列表路径')
add_arg('scaler_path',      str,    'dataset/standard.m',     '测试数据的数据列表路径')
add_arg('model_path',       str,    'output/models/model.pth',  '模型保存的路径')
args = parser.parse_args()


def evaluate():
    # 获取评估数据
    eval_dataset = CustomDataset(args.test_list_path,
                                 scaler_path=args.scaler_path,
                                 mode='eval',
                                 sr=16000,
                                 chunk_duration=3)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    # 获取分类标签
    with open(args.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        class_labels = [l.replace('\n', '') for l in lines]
    # 获取模型
    device = torch.device("cuda")
    model = Model(num_class=args.num_class)
    model.to(device)
    # 加载模型
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    print('开始评估...')
    # 开始评估
    accuracies, preds, labels = [], [], []
    for batch_id, (audio, label) in enumerate(eval_loader):
        audio = audio.to(device)
        output = model(audio)
        # 计算准确率
        output = output.data.cpu().numpy()
        # 模型预测标签
        pred = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))
        preds.extend(pred)
        # 真实标签
        labels.extend(label.tolist())
        # 准确率
        accuracies.append(acc)
    acc = float(sum(accuracies) / len(accuracies))
    cm = confusion_matrix(labels, preds)
    print('分类准确率: {:.4f}'.format(acc))
    plot_confusion_matrix(cm=cm, save_path='output/log/混淆矩阵_eval.png', class_labels=class_labels, show=False)


if __name__ == '__main__':
    print_arguments(args)
    evaluate()
