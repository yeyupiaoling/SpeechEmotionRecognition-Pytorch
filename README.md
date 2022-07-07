# 语音情感识别
本项目是基于Pytorch实现的语音情感识别，效果一般，提供给大家参考学习。

 - Pytorch 1.10.0

# 使用

1. 准备数据集，语音数据集放在`dataset/audios`，每个文件夹存放一种情感的语音，例如`dataset/audios/angry/`、`dataset/audios/fear/`等等，然后执行下面命令生成数据列表。
```shell
pycreate_data.py
```

2. 开始训练，其他参数不重要，最重要的是`num_class`分类类别大小，要根据自己的分类数量来修改。
```shell
python train.py
```

3. 评估模型，同样要修改`num_class`。
```shell
python eval.py
```

4. 导出模型，用于预测部署，记得要修改`num_class`。
```shell
python export_model.py
```

5. 预测语音文件。
```shell
python infer.py --audio_path=dataset/audios/angry/audio_0.wav
```

# 数据预处理
在语音情感识别中，我首先考虑的是语音的数据预处理，按照声音分类的做法，本人一开始使用的是声谱图和梅尔频谱。声谱图和梅尔频谱这两种数据预处理在声音分类中有着非常好的效果，具体的预处理方式如下，但是效果不佳，所以改成本项目使用的预处理方式，这个种预处理方式是使用多种处理方式合并在一起的。

1. 声谱图数据预处理方式。
```python
linear = librosa.stft(wav, n_fft=400, win_length=400, hop_length=160)
features, _ = librosa.magphase(linear)
features = librosa.power_to_db(features, ref=1.0, amin=1e-10, 
top_db=None)
mean = np.mean(features, 0, keepdims=True)
std = np.std(features, 0, keepdims=True)
features = (features - mean) / (std + 1e-5)
```

2. 梅尔频谱数据预处理方式
```python
wav, sr_ret = librosa.load(audio_path, sr=16000)
features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, 
n_mels=80, hop_length=160, win_length=400)
features = librosa.power_to_db(features, ref=1.0, amin=1e-10, 
top_db=None)
mean = np.mean(features, 0, keepdims=True)
std = np.std(features, 0, keepdims=True)
features = (features - mean) / (std + 1e-5)
```

# 模型
在模型结构上，一开始使用ECAPA-TDNN 模型结构，效果也不佳，变改成本项目的模型结构，然后经过多次测试，发现把该模型上的LSTM层改为双向的，效果会更佳。同时为了提高模型的拟合能力，也把每层的大小都提高了，结构如下。

```python
class Model(nn.Layer):
    def __init__(self, num_class):
        super().__init__()
        self.fc0 = nn.Linear(in_features=312, out_features=512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, direction='bidirect')
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=num_class)

    def forward(self, x):
        x = self.fc0(x)
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        y, (h, c) = self.lstm(x)
        x = y.squeeze(axis=1)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
```

# 数据增强
发现在数据预处理和模型结构上修改有所进展之后，本有变开始对数据增强入手，希望通过数据增强方式提高模型的准确率。本项目添加了随机增强，噪声增强、音量增强、语速增强这四种增强方式，可以通过配置文件`configs/augment.yml`修改增强方式。
```yaml
noise:
  min_snr_dB: 10
  max_snr_dB: 30
  noise_path: "dataset/noise"
  prob: 0.0

speed:
  min_speed_rate: 0.9
  max_speed_rate: 1.1
  num_rates: 3
  prob: 0.5

volume:
  min_gain_dBFS: -15
  max_gain_dBFS: 15
  prob: 0.5

```
