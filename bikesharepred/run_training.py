import argparse
from pathlib import Path

import torch_geometric.loader as geomloader

from config import Config
import trainer
from dataset import BikeGraphDataset

import torch as th
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='default')
    args = parser.parse_args()

    np.random.seed(42)
    th.manual_seed(42)

    cfg = Config.load_from_json(args.config_file)
    dataset = BikeGraphDataset(cfg)
    dataset.process()
    cfg._calculate_dependent_params()
    

    train, val, test = dataset.get_day_splits(train_frac=0.7, val_frac=0.15)
    model = trainer.model_train(train, val, test, cfg)




