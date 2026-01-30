import numpy as np


def mask_timeseries_with_random_square(timeseries, min_size=8, max_size=32):
    """
    Applies a random square mask to each image in a time series.

    Parameters:
      timeseries (np.ndarray): Array of shape (T, H, W)
      min_size (int): Minimum side length of the square.
      max_size (int): Maximum side length of the square.

    Returns:
      masked_timeseries (np.ndarray): Array with the square regions masked (set to zero).
      masks (np.ndarray): Boolean mask array of the same shape, where False indicates masked pixels.
    """
    T, H, W = timeseries.shape

    # Generate random centroids and square sizes for each frame.
    centroids_x = np.random.randint(0, H, size=T)
    centroids_y = np.random.randint(0, W, size=T)
    square_sizes = np.random.randint(min_size, max_size + 1, size=T)
    half_sizes = square_sizes // 2

    # Compute top, left, bottom, and right boundaries for each frame.
    tops = np.clip(centroids_x - half_sizes, 0, H)
    lefts = np.clip(centroids_y - half_sizes, 0, W)
    bottoms = np.clip(tops + square_sizes, 0, H)
    rights = np.clip(lefts + square_sizes, 0, W)

    # Create a coordinate grid for one frame.
    X = np.arange(H).reshape(H, 1)  # shape (H, 1)
    Y = np.arange(W).reshape(1, W)  # shape (1, W)

    # Reshape boundaries to (T, 1, 1) for broadcasting.
    tops = tops.reshape(T, 1, 1)
    lefts = lefts.reshape(T, 1, 1)
    bottoms = bottoms.reshape(T, 1, 1)
    rights = rights.reshape(T, 1, 1)

    # Create mask: For each frame, pixels inside the square become False; outside are True.
    mask = ~((X >= tops) & (X < bottoms) & (Y >= lefts) & (Y < rights))  # shape (T, H, W)

    # Apply the mask to each frame.
    masked_timeseries = timeseries * mask.astype(timeseries.dtype)

    return masked_timeseries, mask


def mask_los_disp(y, thresh=0.1, return_norm=False, eps=1e-8):
    """Displacement smaller than 1mm (thresh) is automatically cut off, as undetectable."""
    min_y, max_y = np.min(y, axis=(2, 3), keepdims=True), np.max(y, axis=(2, 3), keepdims=True)
    safe_max_y = np.where(np.abs(max_y) < eps, eps, max_y)
    safe_min_y = np.where(np.abs(min_y) < eps, eps, min_y)
    norm_y = np.where(y > 0, y / np.abs(safe_max_y), y / np.abs(safe_min_y))
    # norm_y = np.where(y > 0, y / np.abs(max_y), y / np.abs(min_y))
    mask_y = (np.abs(norm_y) > 0.5).astype(int).astype(np.float32)
    mask_y[np.abs(y) < thresh] = 0.
    if return_norm:
        return mask_y, norm_y
    else:
        return mask_y


def mask_los_disp_highest_disp(y, frac=.5, thresh=0.1, return_norm=False, eps=1e-8):
    """Displacement smaller than 1mm (thresh) is automatically cut off, as undetectable."""
    max_y = np.max(np.abs(y), axis=(2, 3), keepdims=True)
    safe_max_y = np.where(np.abs(max_y) < eps, eps, max_y)
    norm_y = y / np.abs(safe_max_y)
    mask_y = (np.abs(norm_y) > frac).astype(int).astype(np.float32)
    mask_y[np.abs(y) < thresh] = 0.
    if return_norm:
        return mask_y, norm_y
    else:
        return mask_y
