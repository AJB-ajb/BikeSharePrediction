import torch as th
import torch.nn as nn
import numpy as np
from dataset import z_score, un_z_score, RMSE, MAE, MAPE
import tqdm
import torch_geometric.loader as geomloader

from config import Config
from st_gat import STGAT
from linear_model import LinearModel
from dataset import BikeGraphDataset

import plotting

def finite_diff_third_derivative(input_tensor, axis=-1, step_size=1.0):
    """
    Compute the fourth order central finite difference approximation of the third derivative of an array along a given axis.
    

    Args:
        input_tensor (torch.Tensor): The input tensor.
        axis (int): The axis along which to compute the derivative.
        step_size (float): The spacing between points in the grid.

    Returns:
        torch.Tensor: Tensor with the third derivative along the specified axis.
    """
    if step_size <= 0:
        raise ValueError("Step size must be positive.")
    
    # Shift tensors for central difference
    # note: shifted minus three at i gives the element i+3, thus goes against the stencil convention and we need an additional minus in the stencil
    shifted_minus3 = th.roll(input_tensor, shifts=-3, dims=axis)
    shifted_minus2 = th.roll(input_tensor, shifts=-2, dims=axis)
    shifted_minus1 = th.roll(input_tensor, shifts=-1, dims=axis)
    shifted_plus1 = th.roll(input_tensor, shifts=1, dims=axis)
    shifted_plus2 = th.roll(input_tensor, shifts=2, dims=axis)
    shifted_plus3 = th.roll(input_tensor, shifts=3, dims=axis)
    
    stencil = -th.tensor([1/8, -1, 13/8, 0, -13/8, 1, -1/8])
    stencil = stencil.to(input_tensor.device)

    third_derivative = (shifted_minus3 * stencil[0] + shifted_minus2 * stencil[1] + shifted_minus1 * stencil[2] + shifted_plus1 * stencil[4] + shifted_plus2 * stencil[5] + shifted_plus3 * stencil[6]) / (step_size ** 3)
    
    return third_derivative

def average(dcts):
    "Calculate the average of each key in a list of dictionaries and return as dict."
    return {key: sum(dct[key] for dct in dcts) / len(dcts) for key in dcts[0].keys()}

def loss_fn(y_rate_pred, y_demand_pred, y_truth, y_mask, cfg):
    """
        The loss for the combined rate and demand prediction model.
        y_rate_pred, y_demand_pred, y_truth, y_mask are all of shape [batch_size × (N_predictions * 2)].
    """
    
    rate_error = th.mean((y_rate_pred - y_truth) ** 2)
    # the demand loss when mask = 1 i.e. when the stations capacity is not at its relative max or min
    demand_violation = th.mean((y_demand_pred - y_truth) ** 2 * y_mask)
    # if the demand is at its max or min, the demand must be higher than the relative truth
    demand_violation += th.mean(th.relu(y_truth - y_demand_pred)**2 * (~y_mask))

    # add additional loss term to penalize if demand or rates are lower than 0
    other_components = th.tensor(0., device = y_rate_pred.device)

    if cfg.negative_penalty_factor is not None:
        negative_penalty = cfg['negative_penalty_factor'] * th.mean((y_rate_pred < 0) * y_rate_pred ** 2 + (y_demand_pred < 0) * y_demand_pred ** 2)
        other_components += negative_penalty
    
    # penalize the third derivative of the demand
    α = cfg.third_derivative_penalty or th.tensor(0., device = y_rate_pred.device)
    smoothness_violation = th.tensor(0.)
    if cfg.third_derivative_penalty is not None:
        smoothness_violation = th.mean(finite_diff_third_derivative(y_demand_pred, axis=1, step_size=cfg.subsample_minutes) ** 2)


    return rate_error + demand_violation + α * smoothness_violation + other_components, {'RRateMSE': th.sqrt(rate_error), 'RDemMSViol': th.sqrt(demand_violation), 'RSmoothViol': th.sqrt(smoothness_violation)}

def calculate_metrics(y_rate_pred, y_demand_pred, y_truth, cfg, batch):
    """
        Calculate a set of metrics for rate and demand predictions and return as dict.
        params:
            y_shape: the shape of the tensor that the loss function expects.
        Note: takes in the unnormalized values.
    """
    y_shape = batch.y.shape
    _, loss_components = loss_fn(y_rate_pred.reshape(y_shape), y_demand_pred.reshape(y_shape), y_truth.reshape(y_shape), batch.y_mask, cfg)

    mse = nn.functional.mse_loss(y_rate_pred, y_truth)
    rmse = RMSE(y_truth, y_rate_pred)
    mae = MAE(y_truth, y_rate_pred)

    Δdemand_rate = (y_demand_pred - y_rate_pred).abs().mean()
    
    # --------------- compute averaged mse for each step in the future separately ----------------
    mse_per_future_step = (y_rate_pred - y_truth).pow(2).mean(dim = (0, 2))
    return loss_components | {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MSE per step': mse_per_future_step, 'MA Demand Rate Difference': Δdemand_rate}
    
    
from torch.utils.tensorboard import SummaryWriter
@th.no_grad()
def eval(model, dataset, dataloader, cfg):
    """
        Evaluate the model on the given data, giving loss, rates MSE (quasi-loss), RMSE, MAE. 
        Return evals_list = [MSE, RMSE, …], evals_dict = {'MSE': MSE, …}, predicted rate values ŷ_rate, predicted demand values ŷ_demand, true values y.
    """
    device = next(model.parameters()).device
    model.eval()
    
    mse, rmse, mae = 0., 0., 0.
    mse_per_future_step = th.zeros(cfg.N_predictions, dtype=th.float32, device=device)

    metrics = [] # other metrics from the loss function

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        y_shape = batch.y.shape
        batch_size = y_shape[0] // cfg.N_stations
        y_truth_rel = batch.y.reshape(batch.y.shape[0], cfg.N_predictions, 2) #.view(y_pred_rel.shape) (unnecessary (?))
        y_pred_rel = model.forward(batch).reshape(batch.y.shape[0], cfg.N_predictions, 4)
        y_rate_pred_rel = y_pred_rel[:, :, :2]
        y_demand_pred_rel = y_pred_rel[:, :, 2:]
        if i == 0:
        # both [num_batches_total × (batch_size * num_nodes) × (num predictions * features per ob ∈ {2, 4})] ; (?)
            y_rate_preds = th.empty((len(dataloader), y_shape[0], cfg.N_predictions, 2),dtype=y_pred_rel.dtype, device=device)
            y_demand_preds = th.empty_like(y_rate_preds)
            y_truths = th.empty_like(y_rate_preds)
            
        loss, _ = loss_fn(y_rate_pred_rel.reshape(y_shape), y_demand_pred_rel.reshape(y_shape), batch.y, batch.y_mask, cfg)

        y_rate_pred = un_z_score(y_rate_pred_rel, dataset.μ, dataset.σ)
        y_demand_pred = un_z_score(y_demand_pred_rel, dataset.μ, dataset.σ)
        y_truth = un_z_score(y_truth_rel, dataset.μ, dataset.σ)

        y_rate_preds[i, :y_pred_rel.shape[0], :, :] =  y_rate_pred
        y_demand_preds[i, :y_pred_rel.shape[0], ...] = y_demand_pred
        y_truths[i, :y_pred_rel.shape[0], ...] = y_truth

        metrics.append(calculate_metrics(y_rate_pred, y_demand_pred, y_truth, cfg, batch) | {'Loss': loss})
    
    averaged_metrics = average(metrics)
    return None, averaged_metrics, y_rate_preds, y_demand_preds, y_truths # first argument unused (legacy)

def instantiate_optimizer(model, cfg):
    optimizer_name = cfg['optimizer']
    optimizer_params = cfg['optimizer_params']
    optim = getattr(th.optim, optimizer_name)
    return optim(model.parameters(), **optimizer_params)    

def instantiate_model(cfg):
    if cfg.model == 'STGAT':
        model = STGAT(cfg = cfg, **cfg.__dict__)
    elif cfg.model == 'LinearModel':
        model = LinearModel(cfg)
    else:
        raise NotImplementedError(f"Model {cfg.model} not implemented")
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.compile()
    return model

def load_model(cfg : Config, checkpoint = None):
    model = instantiate_model(cfg)
    model_path = cfg.model_path(checkpoint or cfg['max_iterations'])
    model.load_state_dict(th.load(model_path))
    return model

def fit_and_evaluate(cfg):
    dataset = BikeGraphDataset(cfg)
    train_dataset, val_dataset, test_dataset = dataset.get_day_splits(train_frac=0.7, val_frac=0.15)

    model = instantiate_model(cfg)
    if cfg.model == 'STGAT':
        model = model_train(train_dataset, val_dataset, test_dataset, cfg)

    elif cfg.model == 'LinearModel':
        model.train(train_dataset)

        test_dataloader = geomloader.DataLoader(test_dataset, batch_size = 32, shuffle = False)

        evals_list, evals_dict, y_rate_preds, y_demand_preds, y_truths = eval(model, test_dataset, test_dataloader, cfg)
        scalars = {key: val.item() for key, val in evals_dict.items() if val.dim() == 0}
        metrics_str = ', '.join([f"{key}: {val:.5e}" for key, val in scalars.items()])
        print(f"Linear model test: {metrics_str}")


def model_train(train_dataset, val_dataset, test_dataset, cfg):
    model = instantiate_model(cfg)
    device = next(model.parameters()).device

    print("Model size in MB:", sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024)

    optimizer = instantiate_optimizer(model, cfg)
    writer = SummaryWriter(log_dir=cfg['log_dir'], comment=cfg['name'])
    
    # log the configuration as hyperparameters to tensorboard

    train_dataloader = geomloader.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_dataloader = geomloader.DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    test_dataloader = geomloader.DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    val_losses = []
    val_mses = []

    iteration = 0
    # init tqdm progress bar
    pbar = tqdm.tqdm(total=cfg['max_iterations'])

    while iteration < cfg['max_iterations']:
        train_losses_epoch = 0
        model.train()
        for batch in train_dataloader:
            # --------------- train ----------------
            optimizer.zero_grad()
            y_pred = model.forward(batch.to(device)).view(batch.y.shape[0], cfg.N_predictions, 4)
            # get the rate and demand predictions
            y_rate_pred = y_pred[:, :, :2].reshape(batch.y.shape)
            y_demand_pred = y_pred[:, :, 2:].reshape(batch.y.shape)
            
            loss, _ = loss_fn(y_rate_pred, y_demand_pred, batch.y, batch.y_mask, cfg)
            
            writer.add_scalar('Loss/train', loss, iteration)
            loss.backward()
            optimizer.step()
            train_losses_epoch += loss.item()
            
            iteration += 1
            pbar.update(1)
            # --------------- evaluate ----------------
            if iteration % cfg['eval_interval'] == 0 or iteration == cfg['max_iterations']:

                model.eval()
                evals_list, evals_dict, y_rate_preds, y_demand_preds, y_truths = eval(model, val_dataset, val_dataloader, cfg)
                scalars = {key: val.item() for key, val in evals_dict.items() if val.dim() == 0}
                metrics_str = ', '.join([f"{key}: {val:.5e}" for key, val in scalars.items()])
                
                print(f"Iter {iteration} train:{train_losses_epoch / len(train_dataloader)} val: {metrics_str}")

                mse_per_future_step = evals_dict.pop('MSE per step')
                for i, mse in enumerate(mse_per_future_step):
                    writer.add_scalar(f'MSE/eval/step_{i}', mse, iteration)
                for key, value in evals_dict.items():
                    writer.add_scalar(f'{key}/eval', value, iteration)

                val_losses.append(evals_dict['Loss'])
                val_mses.append(evals_dict['MSE'])

                i_station = 4
                _plot = plotting.plot_station_over_time(y_rate_preds.cpu(), y_demand_preds.cpu(), y_truths.cpu(), i_station, cfg)
                writer.add_figure(f'Station {i_station}', _plot, global_step=iteration)

                _plot = plotting.plot_horizon_accuracies(mse_per_future_step.cpu().numpy(), cfg)
                writer.add_figure('Horizon accuracies', _plot, global_step=iteration)

                horizon = 4
                _plot = plotting.plot_station_over_time_reg(y_rate_preds.cpu().numpy(), y_demand_preds.cpu().numpy(), y_truths.cpu().numpy(), i_station, horizon = horizon, cfg = cfg)
                writer.add_figure(f'Station {i_station} Regular', _plot, global_step=iteration)
                
                model.train()

            # --------------- save model ----------------
            if iteration % cfg['save_interval'] == 0 or iteration == cfg['max_iterations']:
                th.save(model.state_dict(), cfg.model_path(iteration))

    test_metrics, test_metrics_dict, y_rate_preds, y_demand_preds, y_truths = eval(model, test_dataset, test_dataloader, cfg)
    scalars = {key: val.item() for key, val in test_metrics_dict.items() if val.dim() == 0}
    metrics_str = ', '.join([f"{key}: {val:.5e}" for key, val in scalars.items()])
    
    print(f"Final test metrics: {metrics_str}")

    hparam_dict = {key: val for key, val in cfg.__dict__.items() if isinstance(val, (int, float, str))}
    hparam_dict.update(cfg['optimizer_params'])
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=
                       {'final_test_Loss': test_metrics_dict['Loss'], 
                        'final_test_MSE': test_metrics_dict['MSE'],
                        'final_test_RMSE': test_metrics_dict['RMSE'], 'final_test_MAE': test_metrics_dict['MAE'], 'final_eval_loss': evals_dict['Loss'], 'final_eval_RMSE': evals_dict['RMSE'], 'final_eval_MAE': evals_dict['MAE'], 
                        'final_eval_MSE': evals_dict['MSE'], 
                        'min_val_loss': min(val_losses),
                        'min_val_mse': min(val_mses)})
    # calculate minimum validation mse and minimum validation loss

    writer.close()
    return model