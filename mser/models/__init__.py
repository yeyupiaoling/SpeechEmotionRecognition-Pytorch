import importlib

from loguru import logger
from .base_model import BaseModel
from .bi_lstm import BiLSTM

__all__ = ['build_model']


def build_model(input_size, configs):
    use_model = configs.model_conf.get('model', 'BiLSTM')
    model_args = configs.model_conf.get('model_args', {})
    mod = importlib.import_module(__name__)
    model = getattr(mod, use_model)(input_size=input_size, **model_args)
    logger.info(f'成功创建模型：{use_model}，参数为：{model_args}')
    return model
