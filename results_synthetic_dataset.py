import h5py
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from config.denoiser_params import denoiser_configuration
from dataset.h5insardataset import H5InSARDataset
from utils.metrics_utils import err_as_function_of_snr, signal_to_noise_ratio_2d, \
    err_as_function_of_snr_2d

SMALL_SIZE = 14 + 4
LEGEND_SIZE = 14 + 4
MEDIUM_SIZE = 16 + 4
LARGE_SIZE = 20 + 4
FONT_FAMILY = 'Helvetica'  # 'Helvetica'

style_dict = {
    'font.size': SMALL_SIZE,
    'font.family': FONT_FAMILY,
    'axes.titlesize': SMALL_SIZE,
    'axes.labelsize': MEDIUM_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.titlesize': LARGE_SIZE
}

plt.rcParams.update(style_dict)


def compute_rmse_by_case(y_true, y_pred, verbose=True):
    """
    Compute error metrics separately for negative cases (y_true == 0)
    and positive cases (y_true != 0).

    Returns both traditional (RMSE, std) and robust (median error, MAD) statistics.

    Parameters:
    -----------
    y_true : array
        Ground truth values, shape (n, ...)
    y_pred : array
        Predicted values, shape (n, ...)
    verbose : bool
        If True, print results

    Returns:
    --------
    dict with keys 'negative', 'positive', 'global', each containing:
        - 'rmse': Root Mean Squared Error
        - 'std': Standard deviation of squared errors
        - 'median_error': Median absolute error
        - 'mad': Median Absolute Deviation of absolute errors
        - 'count': Number of samples
    """
    errors = y_true - y_pred

    # Compute squared errors and absolute errors for each sample
    se = np.mean(errors ** 2, axis=(1, 2, 3))  # Squared error per sample
    ae = np.mean(np.abs(errors), axis=(1, 2, 3))  # Absolute error per sample

    # Identify negative cases (no signal) and positive cases (has signal)
    has_signal = np.any(y_true != 0, axis=(1, 2, 3))
    negative_indices = ~has_signal
    positive_indices = has_signal

    results = {}

    # Helper function to compute all metrics
    def compute_metrics(se_subset, ae_subset, label, count):
        # Traditional statistics
        mse = np.mean(se_subset)
        rmse = np.sqrt(mse)
        sigma_se = np.sqrt(np.mean((se_subset - rmse ** 2) ** 2))

        # Robust statistics
        median_error = np.median(ae_subset)
        mad = np.median(np.abs(ae_subset - median_error))

        metrics = {
            'rmse': rmse,
            'std': sigma_se,
            'median_error': median_error,
            'mad': mad,
            'count': count
        }

        if verbose:
            print(f'\n{label}:')
            print(f'  RMSE ± std:         {rmse:.6f} ± {sigma_se:.6f} cm')
            print(f'  Median ± MAD:       {median_error:.6f} ± {mad:.6f} cm')
            print(f'  Sample count:       {count}')

        return metrics

    # Negative cases (y_true == 0)
    if np.sum(negative_indices) > 0:
        results['negative'] = compute_metrics(
            se[negative_indices],
            ae[negative_indices],
            'Negative cases (y_true == 0)',
            np.sum(negative_indices)
        )
    else:
        results['negative'] = None
        if verbose:
            print('\nNo negative cases found')

    # Positive cases (y_true != 0)
    if np.sum(positive_indices) > 0:
        results['positive'] = compute_metrics(
            se[positive_indices],
            ae[positive_indices],
            'Positive cases (y_true != 0)',
            np.sum(positive_indices)
        )
    else:
        results['positive'] = None
        if verbose:
            print('\nNo positive cases found')

    # Global metrics
    results['global'] = compute_metrics(
        se,
        ae,
        'Global (all cases)',
        len(se)
    )

    return results


if __name__ == '__main__':
    max_test_samples = 100_000
    denoiser_configuration.training.train_data_ratio = 0.
    denoiser_configuration.training.val_data_ratio = 0.
    dataset = H5InSARDataset(denoiser_configuration.data.dataset_path)

    n_samples = denoiser_configuration.training.num_train_samples
    ind_test = int(
        n_samples * (denoiser_configuration.training.train_data_ratio + denoiser_configuration.training.val_data_ratio))
    print('Loading data...')
    indices = list(range(ind_test, min(n_samples, ind_test + max_test_samples)))
    subset = Subset(dataset, indices)

    '''samples = [subset[i] for i in range(len(subset))]
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)  # trick to load entire subset in one batch
    (x_batch, _), (_, y_batch) = next(iter(loader))
    x_batch = x_batch.numpy()
    y_batch = y_batch.numpy()'''

    batch_size = 128
    x_batch = np.empty((max_test_samples, 9, 128, 128, 1), dtype=np.float32)
    y_batch = np.empty((max_test_samples, 9, 128, 128, 1), dtype=np.float32)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True,
                        prefetch_factor=2)
    samples_loaded = 0
    start = 0
    for (x_b, _), (_, y_b) in loader:
        b = x_b.shape[0]
        end = start + b

        x_batch[start:end] = x_b.numpy()
        y_batch[start:end] = y_b.numpy()

        samples_loaded += b
        start = end

        if samples_loaded % 1024 == 0:
            progress = 100 * samples_loaded / max_test_samples
            print(f'{samples_loaded}/{max_test_samples} ({progress:.1f}%) samples loaded...')

    print(f'{samples_loaded}/{max_test_samples} ({100 * samples_loaded / max_test_samples:.1f}%) samples loaded.')

    # loading predictions
    print('loading predictions...')
    with h5py.File(denoiser_configuration.model.inference_filename, 'r') as file:
        pred = file['predictions'][:max_test_samples]  # [:n_samples - ind_test][:max_test_samples]

    x_batch = x_batch.squeeze()
    y_batch = y_batch.squeeze()
    pred = pred.squeeze()

    results = compute_rmse_by_case(y_batch, pred)

    print('building 2D SNR...')
    snr = signal_to_noise_ratio_2d(x_batch - y_batch, y_batch)

    # y_batch_tmp = np.repeat(y_batch[:, -1:, ...], y_batch.shape[1], axis=1)
    # pred_tmp = np.repeat(pred[:, -1:, ...], pred.shape[1], axis=1)

    x_2d_mean, y_2d_mean, std_2d_mean, _ = err_as_function_of_snr_2d(y_batch, pred, snr, N_BINS=8, median=False)
    x_2d_median, y_2d_median, std_2d_median, _ = err_as_function_of_snr_2d(y_batch, pred, snr, N_BINS=8, median=True)
    x_all_mean, y_all_mean, std_all_mean, _ = err_as_function_of_snr(y_batch, pred, snr, N_BINS=8, median=False)
    x_all_median, y_all_median, std_all_median, _ = err_as_function_of_snr(y_batch, pred, snr, N_BINS=8, median=True)

    # plt.errorbar(x, y, yerr=std, fmt='o-')

    fig, axis = plt.subplots(1, 1, figsize=(13, 10), dpi=300)

    lw, msize = 2.5, 10.

    for data, color, label in zip([(x_2d_mean, y_2d_mean, std_2d_mean), (x_2d_median, y_2d_median, std_2d_median),
                                   (x_all_mean, y_all_mean, std_all_mean),
                                   (x_all_median, y_all_median, std_all_median)],
                                  ['tab:blue', 'tab:cyan', 'tab:red', 'tab:orange'],
                                  ['Last frame, mean', 'Last frame, median', 'All frames, mean', 'All frames, median']):
        markers, caps, bars = axis.errorbar(data[0], data[1], yerr=data[2], fmt='o-', capsize=2.5, linewidth=lw,
                                            markersize=msize, capthick=1., ecolor=color,
                                            elinewidth=1.5, color=color, alpha=1., label=label)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]

    axis.set_ylabel('Relative error', fontweight='bold', font='DejaVu Sans')
    axis.set_xlabel('signal-to-noise ratio [dB]', fontweight='bold', font='DejaVu Sans')
    plt.yscale('log')
    plt.legend(frameon=False)
    axis.spines[['right', 'top']].set_visible(False)
    # plt.show()
    plt.savefig('/data/giuseppe/insar-denoising/figures/err_vs_snr.pdf', bbox_inches='tight')
    plt.close(fig)
