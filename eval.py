import argparse
import functools
import time

from mser.trainer import MSERTrainer
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/bi_lstm.yml',    "配置文件")
add_arg("use_gpu",          bool,  True,                     "是否使用GPU评估模型")
add_arg('save_matrix_path', str,   'output/images/',         "保存混合矩阵的路径")
add_arg('resume_model',     str,   'models/BaseModel_CustomFeature/best_model/',  "模型的路径")
add_arg('overwrites',       str,    None,    '覆盖配置文件中的参数，比如"train_conf.max_epoch=100"，多个用逗号隔开')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MSERTrainer(configs=args.configs, use_gpu=args.use_gpu, overwrites=args.overwrites)

# 开始评估
start = time.time()
loss, accuracy = trainer.evaluate(resume_model=args.resume_model,
                                  save_matrix_path=args.save_matrix_path)
end = time.time()
print('评估消耗时间：{}s，loss：{:.5f}，accuracy：{:.5f}'.format(int(end - start), loss, accuracy))
