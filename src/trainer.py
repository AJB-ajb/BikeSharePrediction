import torch as th
import torch.nn as nn
import numpy as np
from dataset import z_score, un_z_score, RMSE, MAE, MAPE
import tqdm
from st_gat import STGAT
import os
import matplotlib.pyplot as plt

import torch

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
    shifted_minus3 = torch.roll(input_tensor, shifts=-3, dims=axis)
    shifted_minus2 = torch.roll(input_tensor, shifts=-2, dims=axis)
    shifted_minus1 = torch.roll(input_tensor, shifts=-1, dims=axis)
    shifted_plus1 = torch.roll(input_tensor, shifts=1, dims=axis)
    shifted_plus2 = torch.roll(input_tensor, shifts=2, dims=axis)
    shifted_plus3 = torch.roll(input_tensor, shifts=3, dims=axis)
    
    stencil = -torch.tensor([1/8, -1, 13/8, 0, -13/8, 1, -1/8])
    stencil = stencil.to(input_tensor.device)

    third_derivative = (shifted_minus3 * stencil[0] + shifted_minus2 * stencil[1] + shifted_minus1 * stencil[2] + shifted_plus1 * stencil[4] + shifted_plus2 * stencil[5] + shifted_plus3 * stencil[6]) / (step_size ** 3)
    
    return third_derivative


def loss_fn(y_rate_pred, y_demand_pred, y_truth, y_mask, cfg):
    """
        The loss for the combined rate and demand prediction model.
        y_rate_pred, y_demand_pred, y_truth, y_mask are all of shape [batch_size × (N_predictions * 2)].
    """
    rates_loss = th.mean((y_rate_pred - y_truth) ** 2)
    # the demand loss when mask = 1 i.e. when the stations capacity is not at its relative max or min
    demand_loss = th.mean((y_demand_pred - y_truth) ** 2 * y_mask)
    # if the demand is at its max or min, the demand must be higher than the relative truth
    demand_loss += th.mean(th.relu(y_truth - y_demand_pred)**2 * (~y_mask))

    # add additional loss term to penalize if demand or rates are lower than 0

    if cfg.negative_penalty_factor is not None:
        negative_penalty = cfg['negative_penalty_factor'] * th.mean((y_rate_pred < 0) * y_rate_pred ** 2 + (y_demand_pred < 0) * y_demand_pred ** 2)
        demand_loss += negative_penalty
    
    # penalize the third derivative of the demand
    if cfg.third_derivative_penalty is not None:
        demand_loss += cfg.third_derivative_penalty * th.mean(finite_diff_third_derivative(y_demand_pred, axis=1, step_size=cfg.subsample_minutes) ** 2)

    return rates_loss + demand_loss

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
            
        loss = loss_fn(y_rate_pred_rel.reshape(y_shape), y_demand_pred_rel.reshape(y_shape), batch.y, batch.y_mask, cfg)

        y_rate_pred = un_z_score(y_rate_pred_rel, dataset.μ, dataset.σ)
        y_demand_pred = un_z_score(y_demand_pred_rel, dataset.μ, dataset.σ)
        y_truth = un_z_score(y_truth_rel, dataset.μ, dataset.σ)

        y_rate_preds[i, :y_pred_rel.shape[0], :, :] =  y_rate_pred
        y_demand_preds[i, :y_pred_rel.shape[0], ...] = y_demand_pred
        y_truths[i, :y_pred_rel.shape[0], ...] = y_truth

        mse += nn.functional.mse_loss(y_rate_pred, y_truth)
        rmse += RMSE(y_truth, y_rate_pred)
        mae += MAE(y_truth, y_rate_pred)
    N_batches = len(dataloader)
    mse, rmse, mae = mse / N_batches, rmse / N_batches, mae / N_batches
    return [loss, rmse, mae], {'loss':loss, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}, y_rate_preds, y_demand_preds, y_truths
        

def model_train(train_dataloader, val_dataset, val_dataloader, cfg):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = STGAT(N_nodes = cfg['N_stations'], cfg = cfg, **cfg.__dict__).to(device)

    print("Model size in MB:", sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024)

    optimizer = th.optim.Adam(model.parameters(), **(cfg['optimizer_params']))
    writer = SummaryWriter(log_dir=cfg['log_dir'], comment=cfg['name'])

    # log the configuration as hyperparameters to tensorboard
    cfg.log(writer)

    for epoch in tqdm.tqdm(range(cfg['epochs'])):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model.forward(batch.to(device)).view(batch.y.shape[0], cfg.N_predictions, 4)
            # get the rate and demand predictions
            y_rate_pred = y_pred[:, :, :2].reshape(batch.y.shape)
            y_demand_pred = y_pred[:, :, 2:].reshape(batch.y.shape)
            
            
            loss = loss_fn(y_rate_pred, y_demand_pred, batch.y, batch.y_mask, cfg)
            
            writer.add_scalar('Loss/train', loss, epoch)
            loss.backward()
            optimizer.step()
        if epoch % cfg['eval_interval'] == 0 or epoch == cfg['epochs'] - 1:
            evals_list, evals_dict, y_rate_preds, y_demand_preds, y_truths = eval(model, val_dataset, val_dataloader, cfg)
            metrics_str = ', '.join([f"{key}: {val.item():.5e}" for key, val in evals_dict.items()])
            print(f"Epoch {epoch} val: {metrics_str}")

            writer.add_scalars('eval', evals_dict, epoch)

            station_id = 4
            _plot = plot_station_over_time(y_rate_preds.cpu(), y_demand_preds.cpu(), y_truths.cpu(), station_id, cfg)
            writer.add_figure(f'Station {station_id}', _plot, global_step=epoch)
        if epoch % cfg['save_interval'] == 0 or epoch == cfg['epochs'] - 1:
            th.save(model.state_dict(), cfg.model_path(epoch))
        writer.flush()

    writer.add_hparams({}, metric_dict={'final_eval_loss': evals_dict['loss'], 'final_eval_RMSE': evals_dict['RMSE'], 'final_eval_MAE': evals_dict['MAE']})

    writer.close()

def reshape_data4viz(y, cfg):
    """
        Reshape the data for visualization.
        Given
        y_truths: (N_batches, batch_size * N_nodes, 2 * N_preds)
        reshape and transform into (N_batches * batch_size, N_nodes, 2) in order to allow concatenating data for contiguous segments (e.g. full days).
        Note: This only makes sense for not-shuffled batch data.
    """
    N_batches = y.size(0)
    y_squashed = y.view(N_batches * cfg.batch_size, cfg.N_stations, cfg.N_predictions, 2)
    # swap axes such that the predictions are the axis 1
    y_subsampled = y_squashed[::cfg['N_predictions'], :, :, :].swapaxes(1, 2).reshape(-1, cfg.N_stations, 2)
    return y_subsampled

def plot_station_over_time(y_rate_preds, y_demand_preds, y_truths, i_station, cfg, times = np.arange(0, 288)):
    """
        Plot the rate predictions, demand predictions and true values for a given station over the given times.
    """
    y_truths_reshaped = reshape_data4viz(y_truths,cfg)
    y_preds_reshaped = reshape_data4viz(y_rate_preds,cfg)
    y_demand_preds_reshaped = reshape_data4viz(y_demand_preds,cfg)
    
    y_truth_in_station = y_truths_reshaped[:, i_station, 0]
    y_pred_in_station = y_preds_reshaped[:, i_station, 0]
    y_demand_predsin_station = y_demand_preds_reshaped[:, i_station, 0]

    plt.plot(y_truth_in_station[times], label='True Rate')
    plt.plot(y_pred_in_station[times], label='Prediction')
    plt.plot(y_demand_predsin_station[times], label='Demand Prediction')
    plt.legend()
    plt.gca().set(title='Station {}'.format(i_station), xlabel='time [hours]', ylabel='Bike In-Rate')
    plt.xticks(times[::12], times[::12] * cfg['subsample_minutes'] // 60)
    return plt.gcf()




