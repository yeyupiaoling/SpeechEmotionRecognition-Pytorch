import random
import sys

import warnings
from datetime import datetime

import joblib

from data_utils.utils import audio_features

warnings.filterwarnings("ignore")

import librosa
import numpy as np
from torch.utils import data


# 加载并预处理音频
def load_audio(audio_path, mode='train', sr=16000, chunk_duration=3, augmentors=None):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    if mode == 'train':
        # 随机裁剪
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]
            # 对每次都满长度的再次裁剪
            if random.random() > 0.5:
                wav[:random.randint(1, sr // 4)] = 0
                wav = wav[:-random.randint(1, sr // 4)]
        # 数据增强
        if augmentors is not None:
            for key, augmentor in augmentors.items():
                if key == 'specaug': continue
                wav = augmentor(wav)
    elif mode == 'eval':
        # 为避免显存溢出，只裁剪指定长度
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
    # 获取音频特征
    features = audio_features(X=wav, sample_rate=sr)
    return features


# 数据加载器
class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, scaler_path, mode='train', sr=16000, chunk_duration=3, augmentors=None):
        super(CustomDataset, self).__init__()
        # 当预测时不需要获取数据
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.augmentors = augmentors
        self.scaler = joblib.load(scaler_path)

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split('\t')
            # 加载并预处理音频
            features = load_audio(audio_path, mode=self.mode, sr=self.sr,
                                  chunk_duration=self.chunk_duration, augmentors=self.augmentors)
            features = self.scaler.transform([features])
            features = features.squeeze().astype(np.float32)
            return features, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)
