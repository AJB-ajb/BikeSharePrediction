from pathlib import Path
import torch as th
from torch.utils.tensorboard import SummaryWriter
import json

# custom JSON encoder to handle Path objects
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
    
    @staticmethod
    def dumps(obj):
        return json.dumps(obj, cls=PathEncoder)
    
    @staticmethod
    def loads(s):
        return json.loads(s, object_hook=PathEncoder._decode_dict)
    
    @staticmethod
    def _decode_dict(d):
        for key, val in d.items():
            if isinstance(val, str) and '/' in val:
                try:
                    d[key] = Path(val)
                except:
                    pass
        return d

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # the base dir is always set to the base dir of the github repository
        
        # directory for logging and results files, which should usually be kept
        # in log_base_dir, we create a subdirectory for each experiment
        if 'log_base_dir' not in self.__dict__:
            self.log_base_dir = self.base_dir / 'logs'

        for dir in [self.data_dir, self.processed_dir, self.log_base_dir, self.model_dir, self.log_dir, self.config_dir]:
            dir.mkdir(exist_ok=True, parents = True)
    @property
    def base_dir(self):
        return Path(__file__).resolve().parents[1]

    @property
    def data_dir(self):
        return self.base_dir / 'data'

    @property
    def processed_dir(self):
        return self.data_dir / 'processed'

    def __getitem__(self, name: str):
        return self.__getattribute__(name)
    
    def get(self, name: str, default):
        return self.__dict__.get(name, default)
    def __setattr__(self, name: str, value) -> None:
        self.__dict__[name] = value
    def update(self, new_dict: dict) -> None:
        self.__dict__.update(new_dict)

    def __setitem__(self, name: str, value) -> None:
        if not name in self.__dict__.keys():
            raise KeyError(f'Setting unavailable key {name}. Use update() or "." operator instead to add new keys')
        self.__dict__[name] = value
    
    def model_path(self, epoch):
        """
            Return the path to the model file for the given epoch or identifier.
        """
        return self.model_dir / f'{self.name}_{epoch}.pth'

    @staticmethod
    def default_config():
    # full-scale default configuration
        _dict = {
            # ----------- logging and evaluation parameters 
            # one month of data for 5 minute intervals is roughly 8000 examples, so roughly 300 it / epoch
            'eval_interval': 1200, # in iterations 
            'save_interval': 3000,
            'seed' : 42,
            # ----------- data parameters ------------
            'name' : 'default',
            'month': '05',
            'year': '2024',
            'width_mins': 10, # gaussian filter sigma or average width for calculating in and out rates
            'filter': 'gaussian', # 'average' or 'gaussian'
            'N_stations': None, # load all
            'data_id': 'default', # data id allows reloading the same data for different trials

            # ----------- data processing parameters ------------
            'subsample_minutes': 5, # subsample data to 5 minute intervals
            'N_history': 12, # take 12 subsampled data points as history
            'N_predictions': 9, # predict the following 9 data points
            'min_stations_connected': 10,
            'max_dst_meters': 500, # maximum distance between stations to be considered connected in the GCN graph
            'use_time_features': True, # use explicit embeddings of day of week and time of day
            # ----------- hyperparameters ------------
            # ----------- model parameters ------------
            'batch_size': 32,
            'dropout': 0.1,
            'gat_heads': 8,
            'num_gat_layers': 1,
            'lstm_params': {
                'lstm1_hidden_size': 32,
                'lstm2_hidden_size': 128
            },
            'transformer_params': {
                'n_layers': 4,
                'n_heads': 8,
                'd_model': 64,
                'dim_feedforward': 128
            },
            'optimizer': 'AdamW',
            'optimizer_params': {
                'lr': 5e-4,
                'weight_decay': 1e-5
            },
            'max_iterations': 6000,

            'negative_penalty_factor': 0.0, # additional penalty for negative predictions
            'third_derivative_penalty' : 0.5, # penalty factor for (demand''')^2
            'final_module' : 'lstm', # 'lstm' or 'transformer'
            'model': 'STGAT', # 'STGAT' or 'LinearModel'
            'reload_bike_data': False
        } 
        cfg = Config(**_dict)
        return cfg
    
    @staticmethod
    def valid_keys():
        if not hasattr(Config, '_valid_keys'):
            def_config = Config.default_config()
            Config._valid_keys.update(def_config.__dict__.keys())

    @property 
    def in_features_per_node(self):
        return self.N_history * 2
    
    @property
    def out_features_per_node(self):
        return self.N_predictions * 4
    @property
    def N_in_features_per_step_with_global(self):
        "Global in features per step, needed for LSTM, and linear layer"
        if self.N_stations is None:
            raise ValueError("N_stations not yet set")
        
        N_time_features = 4
        N_base = self.N_stations * 2
        return N_base + N_time_features if self.use_time_features else N_base

    @property
    def final_module_params(self):
        if self.final_module == 'lstm':
            return self.lstm_params
        elif self.final_module == 'transformer':
            return self.transformer_params
        else:
            raise NotImplementedError(f'{self.final_module} not implemented')
        
    @property
    def model_dir(self):
        return self.log_base_dir / 'models'
    @property
    def config_dir(self):
        "The directory where configuration files are stored. When a configuration is loaded, the log_base_dir is infered from the position of the configuration file."
        return self.log_base_dir / 'configs'
    @property
    def log_dir(self):
        "The directory where tensorboard logs are stored"
        return self.log_base_dir / self.name
    

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

    def print(self):
        print(f'Configuration {self.name}:')
        for key, val in self.__dict__.items():
            print(f'    {key}: {val}')
        

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
        
        return cfg
    
    def save_to_json(self, file_path = None):
        """
            Save file to JSON. Does not save the base directory or the log and log base directories. These are infered from the position of the config file when loading.
            If none, saves to `config_dir / name.json`
        """
        # don't save the base directory
        # make a dict
        file_path = file_path or self.config_dir / f'{self.name}.json'
        with open(str(file_path), 'w') as f:
            json.dump(self.__dict__, f, cls=PathEncoder)

    @staticmethod
    def load_from_json(file_path):
        """
            We assume that the config file is in log_base_dir / 'configs' and infer the log_base_dir from the position of the config file.
        """
        file_path = Path(file_path)
        log_base_dir = file_path.parents[1]

        with open(str(file_path), 'r') as f:
            config_dict = json.load(f, object_hook=PathEncoder._decode_dict)
        cfg = Config(**(config_dict | {'log_base_dir': log_base_dir}))
        return cfg
    
    @staticmethod
    def load_from_args(args):
        config_dict = vars(args)
        cfg = Config(**config_dict)
        return cfg