from dataset import BikeGraphDataset, z_score, un_z_score, RMSE, MAE, MAPE
from st_gat import STGAT
import analysis as an

import torch as th
import numpy as np
import torch_geometric.loader as geomloader
from pathlib import Path

import os
from trainer import model_train, eval
from torch.utils.tensorboard import SummaryWriter

os.chdir(str(an.BASE_DIR))

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.model_dir = self.base_dir / 'models' 
        self.log_base_dir = self.base_dir / 'logs'

        for dir in [self.data_dir, self.processed_dir, self.model_dir, self.log_base_dir]:
            dir.mkdir(exist_ok=True)

    def __getitem__(self, name: str):
        return self.__dict__[name]
    def get(self, name: str, default):
        return self.__dict__.get(name, default)
    def __setattr__(self, name: str, value) -> None:
        self.__dict__[name] = value
    def update(self, new_dict: dict) -> None:
        self.__dict__.update(new_dict)
    
    def model_path(self, epoch):
        return self.model_dir / f'{self.name}_{epoch}.pth'

    @staticmethod
    def default_config():
    # full-scale default configuration
        _dict = {
            # ----------- logging and evaluation parameters 
            'eval_interval': 5,
            'save_interval': 10,
            'seed' : 42,
            # ----------- data parameters ------------
            'name' : 'default',
            'month': '05',
            'year': '2024',
            'N_stations': None, # load all

            # ----------- data processing parameters ------------
            'subsample_minutes': 5, # subsample data to 5 minute intervals
            'N_history': 12, # take 12 subsampled data points as history
            'N_predictions': 9, # predict the following 9 data points
            'min_stations_connected': 2,
            'use_time_features': True, # use explicit embeddings of day of week and time of day
            # ----------- hyperparameters ------------
            # ----------- model parameters ------------
            'batch_size': 32,
            'dropout': 0.1,
            'gat_heads': 8,
            'lstm_params': {
                'lstm1_hidden_size': 32,
                'lstm2_hidden_size': 128
            },
            'transformer_params': {
                'n_layers': 4,
                'n_heads': 8,
                'd_model': 32,
                'dim_feedforward': 128
            },
            'optimizer': 'Adam', # td
            'optimizer_params': {
                'lr': 5e-4,
                'weight_decay': 1e-5
            },
            'epochs': 60,
            'negative_penalty_factor': 0.0, # additional penalty for negative predictions
            'third_derivative_penalty' : 0.5, # penalty factor for (demand''')^2
            'final_module' : 'lstm' # 'lstm' or 'transformer'; transformer todo
        } 
        cfg = Config(**_dict)

        cfg.reload_bike_data = False
        cfg._calculate_dependent_params()
        # dependent parameters
        return cfg

    def _calculate_dependent_params(self):
        # calculate dependent parameters
        N_in_features_per_step_node = 2 # in and out rates
        N_features_per_out_step_node = 4 # in and out rates and demands

        self.in_features_per_node = N_in_features_per_step_node * self.N_history
        self.out_features_per_node = N_features_per_out_step_node * self.N_predictions # 2 for in and out rates, 2 for in and out demands

        if self.N_stations is not None:
            # calculate global in features per step, needed for LSTM, and linear layer
            self.N_in_features_per_step_with_global = self.N_stations * N_in_features_per_step_node
            if self.use_time_features:
                self.N_in_features_per_step_with_global += 4 # 4 time features, weekday, daytime (sin, cos)
        else:
            print("Warning: N_stations not yet set, dependent parameters are not yet fully computed and have to be updated later")

        self.log_dir = self.log_base_dir / self.name
        if self.final_module == 'lstm':
            self.final_module_params = self.lstm_params
        elif self.final_module == 'transformer':
            self.final_module_params = self.transformer_params
        else:
            raise NotImplementedError(f'{self.final_module} not implemented')

    def log(self, writer: th.utils.tensorboard.SummaryWriter):
        hparams = {key: val for key, val in self.__dict__.items() if isinstance(val, (int, float, str))}
        optim_hparams = {key: val for key, val in self['optimizer_params'].items()}
        hparams.update(optim_hparams)
        writer.add_hparams(hparam_dict=hparams, metric_dict={})

        # self._log(None, self.__dict__, writer)

    def _log(self, key, val, writer: SummaryWriter):
        if isinstance(val, (int, float, str)):
            writer.add_hparams(hparam_dict={key: val}, metric_dict={})
        elif isinstance(val, dict):
            prefix = key + '.' if key is not None else ''
            for sub_key, val in val.items():
                self._log(f'{prefix}{sub_key}', val, writer)
        

    @staticmethod
    def test_config():
        "small scale minimal test configuration; absolute minimum for fastest exception testing"
        # base the test configuration on the default configuration and only override the necessary values
        cfg = Config.default_config()
        cfg.update({
            'name': 'test',
            'N_stations': 5,
            'N_history': 5,
            'N_predictions': 3,
            'optimizer_params': { # naming follows torch conventions here
                'lr': 0.01
            },
            'epochs': 1,
        })
        cfg._calculate_dependent_params()
        return cfg
    @staticmethod
    def overfit_config():
        # test overfitting with small dataset
        cfg = Config.test_config()
        cfg.update({
            'name': 'overfit',
            'dropout': 0.0,
            'optimizer_params': {
                'lr': 1e-2,
                'weight_decay': 0. # no regularization for overfitting test
            },
            'epochs': 40,
            'N_predictions': 9,
            'N_history': 12
            })
        
        cfg._calculate_dependent_params()
        return cfg

if __name__ == '__main__':
    np.random.seed(42)
    th.manual_seed(42)
    
    TEST = True

    if TEST:
        cfg = Config.test_config()
        # cfg.reload_bike_data = True
        # cfg.final_module = 'transformer'

        dataset = BikeGraphDataset(cfg, root=str())
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

    overfit = True
    if overfit:
        cfg = Config.overfit_config()
        cfg.final_module = 'lstm'
        cfg.optimizer_params['lr'] = 5e-4
        dataset = BikeGraphDataset(cfg, root=str())
        dataset.process()
        cfg._calculate_dependent_params()

        train, val, test = dataset.get_day_splits(train_frac=0.7, val_frac=0.15)
        train_dataloader = geomloader.DataLoader(train, batch_size=cfg['batch_size'], shuffle=True)
        test_dataloader = geomloader.DataLoader(test, batch_size=cfg['batch_size'], shuffle=False)
        model = model_train(train_dataloader, val_dataloader = train_dataloader,val_dataset=val, test_dataset = test, test_dataloader = test_dataloader, cfg = cfg)

    default = False
    if default:
        cfg = Config.default_config()
        # cfg.reload_bike_data = True
        dataset = BikeGraphDataset(cfg, root=str())
        dataset.process()
        cfg._calculate_dependent_params()

        train, val, test = dataset.get_day_splits(train_frac=0.7, val_frac=0.15)
        train_dataloader = geomloader.DataLoader(train, batch_size=cfg['batch_size'], shuffle=True)
        val_dataloader = geomloader.DataLoader(val, batch_size=cfg['batch_size'], shuffle=False)
        test_dataloader = geomloader.DataLoader(test, batch_size=cfg['batch_size'], shuffle=False)
        model_train(train_dataloader, val_dataloader = val_dataloader, val_dataset=val, test_dataloader=test_dataloader, test_dataset=test, cfg = cfg)
        # plot actual samples for comparison