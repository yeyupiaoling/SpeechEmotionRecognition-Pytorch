import os

from mser.trainer import MSERTrainer


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


def create_ravdess_list(audio_dir, list_path):
    labels = ["中性", "平静", "快乐", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶"]
    with open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f'{label}\n')
    data_list = {}
    for d in os.listdir(audio_dir):
        actor_dir = os.path.join(audio_dir, d)
        for file in os.listdir(actor_dir):
            path = os.path.join(actor_dir, file)
            emotion_id = int(file.split('-')[2]) - 1
            if emotion_id not in data_list.keys():
                data_list[emotion_id] = [path]
            else:
                data_list[emotion_id].append(path)
    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')

    for k in data_list.keys():
        for i, file in enumerate(data_list[k]):
            file = file.replace('\\', '/')
            if i % 10 == 0:
                f_test.write(f'{file}\t{k}\n')
            else:
                f_train.write(f'{file}\t{k}\n')
    f_test.close()
    f_train.close()


# 生成归一化文件
def create_standard(config_file):
    trainer = MSERTrainer(configs=config_file)
    trainer.get_standard_file()


if __name__ == '__main__':
    # get_data_list('dataset/audios', 'dataset')
    create_ravdess_list('dataset/Audio_Speech_Actors_01-24', 'dataset')
    create_standard('configs/bi_lstm.yml')
