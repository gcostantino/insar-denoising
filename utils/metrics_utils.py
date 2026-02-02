import numpy as np


def signal_to_noise_ratio(noise, target):
    snr = []
    for i in range(target.shape[0]):
        valid_img_indices = np.where(target[i].sum(axis=(1, 2)) != 0)[0]  # images with nonzero displacement
        if valid_img_indices.shape[0] != 0:
            valid_signal = target[i, valid_img_indices, ...]
            valid_noise = noise[i, valid_img_indices, ...]
            # snr.append(np.mean(10 * np.log10(np.sum(valid_signal ** 2, axis=2) / np.sum(valid_noise ** 2, axis=2))))
            snr.append(np.mean(
                10 * np.log10(np.sum(valid_signal ** 2, axis=0) / np.sum(valid_noise ** 2, axis=0))))  # average on time
            ## snr.append(np.mean(10 * np.log10(peak_to_peak_signal ** 2 / peak_to_peak_noise ** 2)))
        else:
            snr.append(np.nan)
    return np.array(snr)


def binned_err(label, prediction, median=False):
    errors = []
    max_disp_array = []
    n_obs = label.shape[0]
    if n_obs == 0:
        return np.nan, np.nan, np.nan
    for i in range(n_obs):
        # valid_img_indices = np.where(label[i].sum(axis=(1, 2, 3)) != 0)[0]  # stations with nonzero displacement
        valid_img_indices = np.where(label[i].sum(axis=(1, 2)) != 0)[0]  # images with nonzero displacement
        max_disp = np.max(np.abs(label[i][valid_img_indices, ...]), axis=0)
        abs_err = (np.median if median else np.mean)(
            np.abs(label[i][valid_img_indices, ...] - prediction[i][valid_img_indices, ...]), axis=0)
        inner_err = abs_err / max_disp
        # inner_err = abs_err / np.abs(static_disp_label)
        err = (np.median if median else np.mean)(inner_err)  # err = 1 / n_time * np.mean(inner_err)
        errors.append(err)
        max_disp_array.append(max_disp)

    if len(errors) == 0:
        return np.nan, np.nan, np.nan

    errors = np.array(errors)
    # Compute center and spread
    if median:
        center = np.nanmedian(errors)
        # Median Absolute Deviation (scaled by 1.4826 for consistency with std)
        mad = np.nanmedian(np.abs(errors - center))
        # spread = 1.4826 * mad  # Scale to match std for normal distribution
        spread = mad
    else:
        center = np.nanmean(errors)
        spread = np.nanstd(errors)

    max_disp_avg = np.nanmean(max_disp_array)

    # return np.nanmean(errors), np.nanstd(errors), np.nanmean(max_disp_array)
    return center, spread, max_disp_avg


def err_as_function_of_snr(y_true, y_pred, x, N_BINS=20, median=False):
    valid_values = np.logical_and(~np.isnan(x), ~np.isinf(x))  # ~np.isnan(x)
    x, y_true, y_pred = x[valid_values], y_true[valid_values], y_pred[valid_values]
    # _, bin_edges = np.histogram(x, bins=N_BINS)
    bin_edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    snr_bin_mean = []
    err_bin_param = []
    err_std = []
    max_disp_bin_mean = []
    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        print(f'Number of elements in bin {i}:', len(idx_bin))
        mean_bin, std_bin, max_disp_bin = binned_err(y_true[idx_bin], y_pred[idx_bin], median=median)
        err_bin_param.append(mean_bin)
        err_std.append(std_bin)
        snr_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
        max_disp_bin_mean.append(max_disp_bin)
    snr_bin_mean, err_bin_param, err_std = np.array(snr_bin_mean), np.array(err_bin_param), np.array(err_std)
    max_disp_bin_mean = np.array(max_disp_bin_mean)

    # For each SNR bin, check the characteristics of events in that bin
    for i in range(len(bin_edges) - 1):
        # Get events in this SNR bin
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        events_in_bin = y_true[mask]  # True labels for events in this bin

        # 1. Check: Are high-SNR events BIGGER?
        max_displacement_per_event = np.max(np.abs(events_in_bin), axis=(1, 2, 3))
        avg_max_disp = np.mean(max_displacement_per_event)

        # 2. Check: Do high-SNR events have more early signal?
        first_frame_signal = np.mean(np.abs(events_in_bin[:, 1, :, :]))
        last_frame_signal = np.mean(np.abs(events_in_bin[:, -1, :, :]))

        #print(f"SNR bin {bin_edges[i]:.1f} to {bin_edges[i + 1]:.1f} dB:")
        #print(f"  Average max displacement: {avg_max_disp:.3f} cm")
        #print(f"  First frame / last frame signal ratio: {first_frame_signal / last_frame_signal:.3f}")
        #print()

    return snr_bin_mean, err_bin_param, err_std, max_disp_bin_mean


'''def signal_to_noise_ratio_2d(noise, target):
    snr = []
    for i in range(target.shape[0]):
        valid_img_indices = np.where(target[i][-1].sum() != 0)[0]  # images with nonzero displacement
        if valid_img_indices.shape[0] != 0:
            valid_signal = target[i, -1, ...]
            valid_noise = noise[i, -1, ...]
            # snr.append(np.mean(10 * np.log10(np.sum(valid_signal ** 2, axis=2) / np.sum(valid_noise ** 2, axis=2))))
            snr.append(np.mean(
                10 * np.log10(np.sum(valid_signal ** 2) / np.sum(valid_noise ** 2))))  # average on time
            ## snr.append(np.mean(10 * np.log10(peak_to_peak_signal ** 2 / peak_to_peak_noise ** 2)))
        else:
            snr.append(np.nan)
    return np.array(snr)'''


def signal_to_noise_ratio_2d(noise, target):
    """
    Vectorized SNR computation using final time step.
    noise, target: shape (n_observations, time, height, width)
    Returns: SNR array of shape (n_observations,)
    """
    # Extract last time step for all observations at once
    signal_final = target[:, -1, :, :]  # (n_observations, height, width)
    noise_final = noise[:, -1, :, :]  # (n_observations, height, width)

    # Compute power for each observation (sum over spatial dimensions)
    signal_power = np.sum(signal_final ** 2, axis=(1, 2))  # (n_observations,)
    noise_power = np.sum(noise_final ** 2, axis=(1, 2))  # (n_observations,)

    # Check which observations have signal
    has_signal = np.sum(signal_final, axis=(1, 2)) != 0  # (n_observations,) boolean

    # Compute SNR, setting invalid cases to NaN
    snr = np.full(target.shape[0], np.nan)
    valid_mask = has_signal & (noise_power > 0)
    snr[valid_mask] = 10 * np.log10(signal_power[valid_mask] / noise_power[valid_mask])

    return snr


def binned_err_2d_old(label, prediction, median=False):
    errors = []
    max_disp_array = []
    n_obs = label.shape[0]
    if n_obs == 0:
        return np.nan, np.nan, np.nan
    for i in range(n_obs):
        # valid_img_indices = np.where(label[i].sum(axis=(1, 2, 3)) != 0)[0]  # stations with nonzero displacement
        valid_img_indices = np.where(label[i][-1].sum() != 0)[0]  # images with nonzero displacement
        max_disp = np.max(np.abs(label[i][-1]))
        abs_err = (np.median if median else np.mean)(np.abs(label[i][-1] - prediction[i][-1]))
        inner_err = abs_err / max_disp
        # inner_err = abs_err / np.abs(static_disp_label)
        err = (np.median if median else np.mean)(inner_err)  # err = 1 / n_time * np.mean(inner_err)
        errors.append(err)
        max_disp_array.append(max_disp)
    return np.nanmean(errors), np.nanstd(errors), np.nanmean(max_disp_array)


def binned_err_2d(label, prediction, median=False, frame_n=-1):
    """
    Compute error using only the last time step.
    """
    errors = []
    max_disp_array = []
    n_obs = label.shape[0]

    if n_obs == 0:
        return np.nan, np.nan, np.nan

    for i in range(n_obs):
        # Check if last frame has nonzero displacement
        if label[i, frame_n, :, :].sum() == 0:  # Fixed: proper indexing and comparison
            continue

        # Extract last frame
        label_last = label[i, frame_n, :, :]
        pred_last = prediction[i, frame_n, :, :]

        # Maximum displacement
        max_disp = np.max(np.abs(label_last))

        # Absolute error (spatial mean or median)
        abs_err = (np.median if median else np.mean)(np.abs(label_last - pred_last))

        # Relative error
        inner_err = abs_err / max_disp

        errors.append(inner_err)
        max_disp_array.append(max_disp)

    if len(errors) == 0:
        return np.nan, np.nan, np.nan

    errors = np.array(errors)

    # Compute center and spread (same logic as binned_err)
    if median:
        center = np.nanmedian(errors)
        mad = np.nanmedian(np.abs(errors - center))
        spread = mad  # Unscaled MAD
    else:
        center = np.nanmean(errors)
        spread = np.nanstd(errors)

    max_disp_avg = np.nanmean(max_disp_array)

    return center, spread, max_disp_avg

def err_as_function_of_snr_2d(y_true, y_pred, x, N_BINS=20, median=False, frame_n=-1):
    valid_values = np.logical_and(~np.isnan(x), ~np.isinf(x))  # ~np.isnan(x)
    x, y_true, y_pred = x[valid_values], y_true[valid_values], y_pred[valid_values]
    # _, bin_edges = np.histogram(x, bins=N_BINS)
    bin_edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    snr_bin_mean = []
    err_bin_param = []
    err_std = []
    max_disp_bin_mean = []
    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        print(f'Number of elements in bin {i}:', len(idx_bin))
        mean_bin, std_bin, max_disp_bin = binned_err_2d(y_true[idx_bin], y_pred[idx_bin], median=median, frame_n=frame_n)
        err_bin_param.append(mean_bin)
        err_std.append(std_bin)
        snr_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
        max_disp_bin_mean.append(max_disp_bin)
    snr_bin_mean, err_bin_param, err_std = np.array(snr_bin_mean), np.array(err_bin_param), np.array(err_std)
    max_disp_bin_mean = np.array(max_disp_bin_mean)
    return snr_bin_mean, err_bin_param, err_std, max_disp_bin_mean


def signal_to_noise_ratio_new(noise, target):
    """
    Compute SNR according to equation (20).
    signal: X(x,y,t) shape (n_observations, time, height, width)
    noise: N(x,y,t) shape (n_observations, time, height, width)
    target: F(x,y,t) shape (n_observations, time, height, width) - ground truth

    Returns SNR in dB for each observation
    """
    snr = []

    for i in range(target.shape[0]):
        # S_i = set of valid pixels: pixels where sum over time is nonzero
        valid_pixel_mask = target[i].sum(axis=0) != 0  # (height, width) boolean mask

        if not np.any(valid_pixel_mask):
            snr.append(np.nan)
            continue

        # Get valid pixels coordinates
        n_valid_pixels = np.sum(valid_pixel_mask)

        # Extract time series for valid pixels only
        F_valid = target[i][:, valid_pixel_mask]  # (time, n_valid_pixels)
        N_valid = noise[i][:, valid_pixel_mask]  # (time, n_valid_pixels)

        # Compute power summed over time for each pixel
        signal_power_per_pixel = np.sum(F_valid ** 2, axis=0)  # (n_valid_pixels,)
        noise_power_per_pixel = np.sum(N_valid ** 2, axis=0)  # (n_valid_pixels,)

        # SNR per pixel in dB
        snr_per_pixel = 10 * np.log10(signal_power_per_pixel / noise_power_per_pixel)

        # Average over all valid pixels (equation 20)
        snr.append(np.mean(snr_per_pixel))

    return np.array(snr)


def err_as_function_of_snr_new(y_true, y_pred, x, N_BINS=20):
    """
    Bin errors by SNR values
    """
    valid_values = np.logical_and(~np.isnan(x), ~np.isinf(x))
    x, y_true, y_pred = x[valid_values], y_true[valid_values], y_pred[valid_values]

    bin_edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    snr_bin_mean = []
    err_bin_param = []
    err_std = []

    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        print(f'Number of elements in bin {i}:', len(idx_bin))

        if len(idx_bin) > 0:
            mean_bin, std_bin = binned_err(y_true[idx_bin], y_pred[idx_bin])
            err_bin_param.append(mean_bin)
            err_std.append(std_bin)
            snr_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))

    snr_bin_mean = np.array(snr_bin_mean)
    err_bin_param = np.array(err_bin_param)
    err_std = np.array(err_std)

    return snr_bin_mean, err_bin_param, err_std


def binned_err_new(label, prediction):
    """
    label, prediction: shape (n_obs, time, height, width)
    Returns mean and std of relative errors
    """
    errors = []
    n_obs = label.shape[0]

    if n_obs == 0:
        return np.nan, np.nan

    for i in range(n_obs):
        # Find time steps with nonzero displacement (summing over spatial dimensions)
        valid_time_indices = np.where(label[i].sum(axis=(1, 2)) != 0)[0]

        if valid_time_indices.shape[0] == 0:
            continue

        # Extract valid time steps
        valid_label = label[i, valid_time_indices, ...]  # (n_valid_time, height, width)
        valid_pred = prediction[i, valid_time_indices, ...]  # (n_valid_time, height, width)

        # Compute absolute error for each time step, averaged over spatial dimensions
        abs_err = np.mean(np.abs(valid_label - valid_pred), axis=(1, 2))  # (n_valid_time,)

        # Normalize by maximum absolute value for each time step
        max_abs_label = np.max(np.abs(valid_label), axis=(1, 2))  # (n_valid_time,)

        # Avoid division by zero
        max_abs_label = np.where(max_abs_label == 0, 1e-10, max_abs_label)

        # Relative error for each time step
        inner_err = abs_err / max_abs_label  # (n_valid_time,)

        # Mean relative error across time steps for this observation
        err = np.mean(inner_err)
        errors.append(err)

    if len(errors) == 0:
        return np.nan, np.nan

    return np.mean(errors), np.std(errors)
