import numpy as np
import matplotlib.pyplot as plt

from config import Config

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

    plt.clf()
    plt.plot(y_truth_in_station[times], label='True Rate')
    plt.plot(y_pred_in_station[times], label='Prediction', marker = 'o')
    plt.plot(y_demand_predsin_station[times], label='Demand Prediction', marker = 'x')
    plt.legend()
    plt.gca().set(title='Station {}'.format(i_station), xlabel='time [hours]', ylabel='Bike In-Rate')
    plt.xticks(times[::12], times[::12] * cfg['subsample_minutes'] // 60)
    return plt.gcf()

def plot_horizon_over_time(ys, i_station, horizon, times, cfg, in_rate, **plot_kwargs):
    def extract_horizon(y : np.ndarray):
        N_batches = y.shape[0]
        y_squashed = y.reshape((N_batches * cfg.batch_size, cfg.      N_stations, cfg.N_predictions, 2))
        ys_future = y_squashed[times, i_station, horizon, :]
        return ys_future
    y_extracted = extract_horizon(ys)

    last_dim_idx = 0 if in_rate else 1
    plt.plot(y_extracted[:, last_dim_idx], **plot_kwargs)


def plot_station_over_time_reg(y_rate_preds, y_demand_preds, y_truths, i_station, horizon, cfg, times = np.arange(0, 288), in_rate = True, **plot_kwargs):
    """
        Plot the rate predictions, demand predictions and true values for a given station over the given times. Use the `horizon` index into the future, i.e. for horizon = 0, the first prediction is used, etc.
    """
    # y_rate_preds: [num_batches_total × (batch_size * num_nodes) × (num predictions * 2)]

    ylabel = 'In Rate' if in_rate else 'Out Rate'

    plt.clf()

    x_times = times * cfg.subsample_minutes

    plot_horizon_over_time(x_times,y_rate_preds, i_station, horizon, times, cfg, in_rate, label='Predicted Rate', linestyle='--')
    plot_horizon_over_time(x_times,y_demand_preds, i_station, horizon, times, cfg, in_rate, label='Predicted Demand', linestyle=':')
    plot_horizon_over_time(x_times, y_truths, i_station, horizon, times, cfg, in_rate, label='True Rate')

    plt.legend()
    horizon_minutes = horizon * cfg.subsample_minutes
    args = {'title':f'Station {i_station} for Horizon {horizon_minutes}', 'xlabel':'Time [Minutes]', 'ylabel': 'Bike ' + ylabel + '[Bikes/Hour]'}
    args |= plot_kwargs
    plt.gca().set(**args)
    
    return plt.gcf()

def plot_horizon_accuracies(mses_per_step, cfg : Config, **plot_kwargs):
    rmses_per_step = np.sqrt(mses_per_step)
    
    subsample_mins = cfg['subsample_minutes']
    minutes = np.arange(1 * subsample_mins, (len(rmses_per_step) + 1) * subsample_mins, subsample_mins)
    
    plt.clf()
    plt.plot(minutes, rmses_per_step)

    args = {'title':'RMSE per Step', 'xlabel':'Minutes into Future', 'ylabel':'RMSE'} | plot_kwargs
    plt.gca().set(**args)
    return plt.gcf()