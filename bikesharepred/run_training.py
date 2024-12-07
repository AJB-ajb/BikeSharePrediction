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


    cfg = Config.load_from_json(args.config_file)
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    dataset = BikeGraphDataset(cfg)
    cfg.save_to_json(None) # to update N_stations
    dataset.process()

    train, val, test = dataset.get_day_splits(train_frac=0.7, val_frac=0.15)
    model = trainer.model_train(train, val, test, cfg)




