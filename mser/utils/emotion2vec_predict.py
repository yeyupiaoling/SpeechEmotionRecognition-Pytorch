# @Time    : 2024-07-02
# @Author  : yeyupiaoling
import os
import shutil

import numpy as np
from funasr import AutoModel
from modelscope import snapshot_download
from loguru import logger


class Emotion2vecPredict(object):
    def __init__(self, model_id, revision, use_gpu=True):
        emotion2vec_model_dir = 'models/'
        save_model_dir = os.path.join(emotion2vec_model_dir, model_id)
        if not os.path.exists(save_model_dir):
            model_dir = snapshot_download(model_id, revision=revision)
            shutil.copytree(model_dir, save_model_dir)
        self.model = AutoModel(model=save_model_dir,
                               log_level="ERROR",
                               device='cuda' if use_gpu else 'cpu',
                               disable_pbar=True,
                               disable_log=True)
        logger.info(f"成功加载模型：{save_model_dir}")

    def extract_features(self, x, kwargs):
        res = self.model.generate(input=[x], **kwargs)
        feats = res[0]["feats"]
        return feats

    def predict(self, audio):
        res = self.model.generate(audio, granularity="utterance", extract_embedding=False)
        labels, scores = [], []
        for result in res:
            label, score = result["labels"], result["scores"]
            lab = np.argsort(score)[-1]
            s = score[lab]
            l = label[lab].split("/")[0]
            labels.append(l)
            scores.append(round(float(s), 5))
        return labels, scores


