import os
import shutil

import librosa
import numpy as np
from loguru import logger


class AudioFeaturizer(object):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

    def __init__(self, feature_method='Emotion2Vec', method_args={}):
        super().__init__()
        self._method_args = method_args
        self._feature_method = feature_method
        self._feature_model = None
        logger.info(f'使用的特征方法为 {self._feature_method}')

    def __call__(self, x, sample_rate: float) -> np.ndarray:
        if self._feature_method == 'CustomFeature':
            return self.custom_features(x, sample_rate)
        elif self._feature_method == 'Emotion2Vec':
            return self.emotion2vec_features(x)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def emotion2vec_features(self, x) -> np.ndarray:
        from mser.utils.emotion2vec_predict import Emotion2vecPredict
        if self._feature_model is None:
            self._feature_model = Emotion2vecPredict('iic/emotion2vec_base', revision="v2.0.4", use_gpu=True)
        feats = self._feature_model.extract_features(x, self._method_args)
        return feats

    @staticmethod
    def custom_features(x, sample_rate: float) -> np.ndarray:
        stft = np.abs(librosa.stft(x))

        # fmin 和 fmax 对应于人类语音的最小最大基本频率
        pitches, magnitudes = librosa.piptrack(y=x, sr=sample_rate, S=stft, fmin=70, fmax=400)
        pitch = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, 1].argmax()
            pitch.append(pitches[index, i])

        pitch_tuning_offset = librosa.pitch_tuning(pitches)
        pitchmean = np.mean(pitch)
        pitchstd = np.std(pitch)
        pitchmax = np.max(pitch)
        pitchmin = np.min(pitch)

        # 频谱质心
        cent = librosa.feature.spectral_centroid(y=x, sr=sample_rate)
        cent = cent / np.sum(cent)
        meancent = np.mean(cent)
        stdcent = np.std(cent)
        maxcent = np.max(cent)

        # 谱平面
        flatness = np.mean(librosa.feature.spectral_flatness(y=x))

        # 使用系数为50的MFCC特征
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
        mfccsstd = np.std(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
        mfccmax = np.max(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)

        # 色谱图
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

        # 梅尔频率
        mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)

        # ottava对比
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

        # 过零率
        zerocr = np.mean(librosa.feature.zero_crossing_rate(x))

        S, phase = librosa.magphase(stft)
        meanMagnitude = np.mean(S)
        stdMagnitude = np.std(S)
        maxMagnitude = np.max(S)

        # 均方根能量
        rmse = librosa.feature.rms(S=S)[0]
        meanrms = np.mean(rmse)
        stdrms = np.std(rmse)
        maxrms = np.max(rmse)

        features = np.array([
            flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
            maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
            pitch_tuning_offset, meanrms, maxrms, stdrms
        ])

        features = np.concatenate((features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast)).astype(np.float32)
        return features

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'CustomFeature':
            return 312
        elif self._feature_method == 'Emotion2Vec':
            return 768
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')
