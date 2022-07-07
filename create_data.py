import os
import joblib
from tqdm import tqdm

from data_utils.reader import load_audio

from sklearn.preprocessing import StandardScaler


# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')

    for i in range(len(audios)):
        f_label.write(f'{audios[i]}\n')
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, audios[i], sound).replace('\\', '/')
            if sound_sum % 10 == 0:
                f_test.write('%s\t%d\n' % (sound_path, i))
            else:
                f_train.write('%s\t%d\n' % (sound_path, i))
            sound_sum += 1
    f_label.close()
    f_train.close()


# 生成归一化文件
def create_standard(list_path, scaler_path):
    with open(os.path.join(list_path, 'train_list.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in tqdm(lines):
        path, label = line.split('\t')
        data.append(load_audio(path, mode='infer'))
    scaler = StandardScaler().fit(data)
    joblib.dump(scaler, scaler_path)


if __name__ == '__main__':
    get_data_list('dataset/audios', 'dataset')
    create_standard('dataset', 'dataset/standard.m')
