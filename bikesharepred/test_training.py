from dataset import BikeGraphDataset, z_score, un_z_score, RMSE, MAE, MAPE
from st_gat import STGAT
import analysis as an

import torch as th
import numpy as np
import torch_geometric.loader as geomloader
from pathlib import Path

import os
from trainer import model_train, eval
import trainer
from torch.utils.tensorboard import SummaryWriter
from config import Config

os.chdir(str(an.BASE_DIR))

if __name__ == '__main__':
    np.random.seed(42)
    th.manual_seed(42)

    TEST_linear = False
    if TEST_linear:
        cfg = Config.default_config()
        cfg.σ = 60
        cfg.model = 'LinearModel'
        # clear gc because fitting the linear model on the whole dataset at once is memory intensive
        import gc
        gc.collect()

        trainer.fit_and_evaluate(cfg)

        os._exit(0)
    
    
    TEST = False

    if TEST:
        cfg = Config.test_config()
        # cfg.reload_bike_data = True
        # cfg.final_module = 'transformer'

        dataset = BikeGraphDataset(cfg)
        dataset.process() # force reprocessing
        cfg._calculate_dependent_params()

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        print("Running on ", device)
        model = STGAT(N_nodes = cfg['N_stations'], cfg = cfg,**cfg.__dict__).to(device)
        dataloader = geomloader.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

        # test the backward pass
        model.train()
        for batch in dataloader:
            result = model.forward(batch.to(device))
            result.sum().backward()
            break
        print("Forward and backward passes work ✔")

    overfit = False
    if overfit:
        cfg = Config.overfit_config()
        cfg.final_module = 'lstm'
        cfg.optimizer_params['lr'] = 5e-4
        dataset = BikeGraphDataset(cfg)
        dataset.process()
        cfg._calculate_dependent_params()

        train, val, test = dataset.get_day_splits(train_frac=0.7, val_frac=0.15)
        model = model_train(train, val, test, cfg)

    default = True
    if default:
        cfg = Config.default_config()
        # cfg.reload_bike_data = True
        dataset = BikeGraphDataset(cfg)
        dataset.process()
        cfg._calculate_dependent_params()

        train, val, test = dataset.get_day_splits(train_frac=0.7, val_frac=0.15)
        model_train(train, val, test, cfg)
        # plot actual samples for comparison