import argparse
import functools

import joblib
import numpy as np
import torch

from data_utils.reader import load_audio
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_path',       str,    'dataset/audios/angry/audio_0.wav',  '需要识别的音频文件')
add_arg('label_list_path',  str,    'dataset/label_list.txt',            '标签列表路径')
add_arg('scaler_path',      str,    'dataset/standard.m',                '测试数据的数据列表路径')
add_arg('model_path',       str,    'output/inference/inference.pth',    '模型保存的路径')
args = parser.parse_args()

# 加载归一化文件
scaler = joblib.load(args.scaler_path)
# 获取分类标签
with open(args.label_list_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
class_labels = [l.replace('\n', '') for l in lines]
# 获取模型
device = torch.device("cuda")
model = torch.load(args.model_path)
model.to(device)
model.eval()


def infer(audio_path):
    data = load_audio(audio_path, mode='infer')
    data = data[np.newaxis, :]
    data = scaler.transform(data)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data).data.cpu().numpy()[0]
    # 显示图片并输出结果最大的label
    lab = np.argsort(output)[-1]
    label = class_labels[lab]
    score = output[lab]
    return label, score


if __name__ == '__main__':
    r, s = infer(audio_path=args.audio_path)
    print(f'识别结果为：{r}，得分：{s}')
