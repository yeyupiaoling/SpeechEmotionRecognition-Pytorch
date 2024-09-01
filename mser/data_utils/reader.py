import joblib
import numpy as np
from torch.utils.data import Dataset
from yeaudio.audio import AudioSegment
from yeaudio.augmentation import ReverbPerturbAugmentor
from yeaudio.augmentation import SpeedPerturbAugmentor, VolumePerturbAugmentor, NoisePerturbAugmentor

from mser.data_utils.featurizer import AudioFeaturizer


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 audio_featurizer:AudioFeaturizer,
                 scaler_path=None,
                 max_duration=3,
                 min_duration=0.5,
                 mode='train',
                 sample_rate=16000,
                 aug_conf=None,
                 use_dB_normalization=True,
                 target_dB=-20):
        """音频数据加载器

        Args:
            data_list_path: 包含音频路径和标签的数据列表文件的路径
            audio_featurizer: 声纹特征提取器
            scaler_path: 归一化文件路径
            max_duration: 最长的音频长度，大于这个长度会裁剪掉
            min_duration: 过滤最短的音频长度
            aug_conf: 用于指定音频增强的配置
            mode: 数据集模式。在训练模式下，数据集可能会进行一些数据增强的预处理
            sample_rate: 采样率
            use_dB_normalization: 是否对音频进行音量归一化
            target_dB: 音量归一化的大小
        """
        super(CustomDataset, self).__init__()
        assert mode in ['train', 'eval', 'create_data', 'extract_feature']
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.speed_augment = None
        self.volume_augment = None
        self.noise_augment = None
        self.reverb_augment = None
        # 获取数据列表
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        if mode == 'train' and aug_conf is not None:
            # 获取数据增强器
            self.get_augmentor(aug_conf)
        # 获取特征器
        self.audio_featurizer = audio_featurizer
        if scaler_path and self.mode != 'create_data':
            self.scaler = joblib.load(scaler_path)

    def __getitem__(self, idx):
        # 分割数据文件路径和标签
        data_path, label = self.lines[idx].replace('\n', '').split('\t')
        # 如果后缀名为.npy的文件，那么直接读取
        if data_path.endswith('.npy'):
            feature = np.load(data_path)
        else:
            # 读取音频
            audio_segment = AudioSegment.from_file(data_path)
            # 数据太短不利于训练
            if self.mode == 'train':
                if audio_segment.duration < self.min_duration:
                    return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
            # 重采样
            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)
            # 音频增强
            if self.mode == 'train':
                audio_segment = self.augment_audio(audio_segment)
            # decibel normalization
            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)
            # 裁剪需要的数据
            if self.mode != 'extract_feature' and audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)
            feature = self.audio_featurizer(audio_segment.samples, sample_rate=audio_segment.sample_rate)
        # 归一化
        if self.mode != 'create_data' and self.mode != 'extract_feature':
            # feature = feature - feature.mean()
            feature = self.scaler.transform([feature])
            feature = feature.squeeze().astype(np.float32)
        return np.array(feature, dtype=np.float32), np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)

    # 获取数据增强器
    def get_augmentor(self, aug_conf):
        if aug_conf.speed is not None:
            self.speed_augment = SpeedPerturbAugmentor(**aug_conf.speed)
        if aug_conf.volume is not None:
            self.volume_augment = VolumePerturbAugmentor(**aug_conf.volume)
        if aug_conf.noise is not None:
            self.noise_augment = NoisePerturbAugmentor(**aug_conf.noise)
        if aug_conf.reverb is not None:
            self.reverb_augment = ReverbPerturbAugmentor(**aug_conf.reverb)

    # 音频增强
    def augment_audio(self, audio_segment):
        if self.speed_augment is not None:
            audio_segment = self.speed_augment(audio_segment)
        if self.volume_augment is not None:
            audio_segment = self.volume_augment(audio_segment)
        if self.noise_augment is not None:
            audio_segment = self.noise_augment(audio_segment)
        if self.reverb_augment is not None:
            audio_segment = self.reverb_augment(audio_segment)
        return audio_segment
