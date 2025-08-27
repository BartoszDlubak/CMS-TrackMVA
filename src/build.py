
# Torch
from frameworks.torch.models.build import build_model
from frameworks.torch.train import TorchTrainer
from frameworks.torch.data_wrapper import TorchDataWrapper

# XGBoost
from frameworks.xgboost.train import XGBTrainer

# # LightGBM
# from frameworks.lightgbm.trainer import LGBMTrainer

# # CatBoost
# from frameworks.catboost.trainer import CatBoostTrainer

# # Sklearn
# from frameworks.sklearn.trainer import SklearnTrainer


def get_trainer(cfg):
    fw = cfg['framework'].lower()

    if fw == 'torch':
        model = build_model(cfg['model'])
        data_wrapper = TorchDataWrapper(**cfg['wrapper'])
        trainer =  TorchTrainer(**cfg['train'])
        trainer.set_model(model)
        trainer.set_wrapper(data_wrapper)
        return trainer
    elif fw == 'xgboost':
        return XGBTrainer(**cfg['train'])
    # elif fw == 'lightgbm':
    #     return LGBMTrainer(**cfg['train'])
    # elif fw == 'catboost':
    #     return CatBoostTrainer(**cfg['train'])
    # elif fw == 'sklearn':
    #     return SklearnTrainer(**cfg['train'])
    else:
        raise ValueError(f'Unknown framework: {fw}')
