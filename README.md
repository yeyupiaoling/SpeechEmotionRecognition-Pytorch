# 基于Pytorch实现的语音情感识别系统

本项目是一个语音情感识别项目，目前效果一般，供大家学习使用。后面会持续优化，提高准确率，如果同学们有好的建议，也欢迎来探讨。

**欢迎大家扫码入知识星球或者QQ群讨论，知识星球里面提供项目的模型文件和博主其他相关项目的模型文件，也包括其他一些资源。**

<div align="center">
  <img src="https://yeyupiaoling.cn/zsxq.png" alt="知识星球" width="400">
  <img src="https://yeyupiaoling.cn/qq.png" alt="QQ群" width="400">
</div>


# 使用准备

 - Anaconda 3
 - Python 3.8
 - Pytorch 1.13.1
 - Windows 10 or Ubuntu 18.04

# 模型测试表

|        模型         | Params(M) | 预处理方法 |   数据集   | 类别数量 | 准确率  |
|:-----------------:|:---------:|:-----:|:-------:|:----:|:----:|
| BidirectionalLSTM |    1.8    | Flank | RAVDESS |  8   | 0.78 |


## 安装环境

 - 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

 - 安装mser库。
 
使用pip安装，命令如下：
```shell
python -m pip install mser -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**建议源码安装**，源码安装能保证使用最新代码。
```shell
git clone https://github.com/yeyupiaoling/SpeechEmotionRecognition-Pytorch.git
cd SpeechEmotionRecognition-Pytorch/
python setup.py install
```

## 准备数据

生成数据列表，用于下一步的读取需要，项目默认提供一个数据集[RAVDESS](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1)，下载这个数据集并解压到`dataset`目录下。

然后执行`create_data.py`里面的`create_ravdess_list('dataset/Audio_Speech_Actors_01-24', 'dataset')`函数即可生成数据列表，同时也生成归一化文件，具体看代码。

```shell
python create_data.py
```

如果自定义数据集，可以按照下面格式，`audio_path`为音频文件路径，用户需要提前把音频数据集存放在`dataset/audio`目录下，每个文件夹存放一个类别的音频数据，每条音频数据长度在3秒左右，如 `dataset/audio/angry/······`。`audio`是数据列表存放的位置，生成的数据类别的格式为 `音频路径\t音频对应的类别标签`，音频路径和标签用制表符 `\t`分开。读者也可以根据自己存放数据的方式修改以下函数。

执行`create_data.py`里面的`get_data_list('dataset/audios', 'dataset')`函数即可生成数据列表，同时也生成归一化文件，具体看代码。
```shell
python create_data.py
```

生成的列表是长这样的，前面是音频的路径，后面是该音频对应的标签，从0开始，路径和标签之间用`\t`隔开。
```shell
dataset/Audio_Speech_Actors_01-24/Actor_13/03-01-01-01-02-01-13.wav	0
dataset/Audio_Speech_Actors_01-24/Actor_01/03-01-02-01-01-01-01.wav	1
dataset/Audio_Speech_Actors_01-24/Actor_01/03-01-03-02-01-01-01.wav	2
```

**注意：** `create_data.py`里面的`create_standard('configs/bi_lstm.yml')`函数必须要执行的，这个是生成归一化的文件。


## 训练

接着就可以开始训练模型了，创建 `train.py`。配置文件里面的参数一般不需要修改，但是这几个是需要根据自己实际的数据集进行调整的，首先最重要的就是分类大小`dataset_conf.num_class`，这个每个数据集的分类大小可能不一样，根据自己的实际情况设定。然后是`dataset_conf.batch_size`，如果是显存不够的话，可以减小这个参数。

```shell
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
# 多卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```


训练输出日志：
```[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:14 - ----------- 额外配置参数 -----------
[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:16 - configs: configs/bi_lstm.yml
[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:16 - local_rank: 0
[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:16 - pretrained_model: None
[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:16 - resume_model: None
[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:16 - save_model_path: models/
[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:16 - use_gpu: True
[2023-08-18 18:48:49.662963 INFO   ] utils:print_arguments:17 - ------------------------------------------------
[2023-08-18 18:48:49.680176 INFO   ] utils:print_arguments:19 - ----------- 配置文件参数 -----------
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:22 - dataset_conf:
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:25 - 	aug_conf:
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		noise_aug_prob: 0.2
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		noise_dir: dataset/noise
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		speed_perturb: True
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		volume_aug_prob: 0.2
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		volume_perturb: False
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:25 - 	dataLoader:
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		batch_size: 32
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		num_workers: 4
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:29 - 	do_vad: False
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:25 - 	eval_conf:
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		batch_size: 1
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:27 - 		max_duration: 3
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:29 - 	label_list_path: dataset/label_list.txt
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:29 - 	max_duration: 3
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:29 - 	min_duration: 0.5
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:29 - 	sample_rate: 16000
[2023-08-18 18:48:49.681177 INFO   ] utils:print_arguments:29 - 	scaler_path: dataset/standard.m
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:29 - 	target_dB: -20
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:29 - 	test_list: dataset/test_list.txt
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:29 - 	train_list: dataset/train_list.txt
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:29 - 	use_dB_normalization: True
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:22 - model_conf:
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:29 - 	num_class: None
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:22 - optimizer_conf:
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:29 - 	learning_rate: 0.001
[2023-08-18 18:48:49.682177 INFO   ] utils:print_arguments:29 - 	optimizer: Adam
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:29 - 	scheduler: WarmupCosineSchedulerLR
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:25 - 	scheduler_args:
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:27 - 		max_lr: 0.001
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:27 - 		min_lr: 1e-05
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:27 - 		warmup_epoch: 5
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:29 - 	weight_decay: 1e-06
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:22 - preprocess_conf:
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:29 - 	feature_method: CustomFeatures
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:22 - train_conf:
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:29 - 	enable_amp: False
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:29 - 	log_interval: 10
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:29 - 	max_epoch: 60
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:29 - 	use_compile: False
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:31 - use_model: BidirectionalLSTM
[2023-08-18 18:48:49.683184 INFO   ] utils:print_arguments:32 - ------------------------------------------------
[2023-08-18 18:48:49.683184 WARNING] trainer:__init__:66 - Windows系统不支持多线程读取数据，已自动关闭！
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
BidirectionalLSTM                        [1, 6]                    --
├─Linear: 1-1                            [1, 512]                  160,256
├─LSTM: 1-2                              [1, 1, 512]               1,576,960
├─Tanh: 1-3                              [1, 512]                  --
├─Dropout: 1-4                           [1, 512]                  --
├─Linear: 1-5                            [1, 256]                  131,328
├─ReLU: 1-6                              [1, 256]                  --
├─Linear: 1-7                            [1, 6]                    1,542
==========================================================================================
Total params: 1,870,086
Trainable params: 1,870,086
Non-trainable params: 0
Total mult-adds (M): 1.87
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 7.48
Estimated Total Size (MB): 7.49
==========================================================================================
[2023-08-18 18:48:51.425936 INFO   ] trainer:train:378 - 训练数据：4407
[2023-08-18 18:48:53.526136 INFO   ] trainer:__train_epoch:331 - Train epoch: [1/60], batch: [0/138], loss: 1.80256, accuracy: 0.15625, learning rate: 0.00001000, speed: 15.24 data/sec, eta: 4:49:49
····················
```

# 评估
每轮训练结束可以执行评估，评估会出来输出准确率，还保存了混合矩阵图片，保存路径`output/images/`，如下。
<br/>
<div align="center">
<img src="docs/images/image1.png" alt="打赏作者" width="600">
</div>


# 预测

在训练结束之后，我们得到了一个模型参数文件，我们使用这个模型预测音频。

```shell
python infer.py --audio_path=dataset/test.wav
```

## 打赏作者
<br/>
<div align="center">
<p>打赏一块钱支持一下作者</p>
<img src="https://yeyupiaoling.cn/reward.png" alt="打赏作者" width="400">
</div>

# 参考资料

1. https://github.com/yeyupiaoling/AudioClassification-Pytorch