import argparse
import functools
import os

import torch

from modules.model import Model
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_class',        int,    6,                                   '分类的类别数量')
add_arg('model_path',       str,    'output/models/model.pth',           '模型保存的路径')
add_arg('save_path',        str,    'output/inference/inference.pth',    '模型保存的路径')
args = parser.parse_args()

# 获取模型
model = Model(num_class=args.num_class)
model.load_state_dict(torch.load(args.model_path))
# 加上Softmax函数
model = torch.nn.Sequential(model, torch.nn.Softmax())

# 保存预测模型
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(model, args.save_path)
