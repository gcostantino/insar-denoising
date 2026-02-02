import torch
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize


def create_fig_time_series(x, pred, y, b_idx):
    """
    Plot a single image time series for TensorBoard logging logic.
    """
    separate_components = (isinstance(pred, list) or isinstance(pred, tuple)) and len(pred) > 2
    time_steps = x.shape[1]
    n_rows, n_cols = 3, 9
    if not separate_components:
        if isinstance(pred, list) or isinstance(pred, tuple):
            mask_pred, pred = pred
            mask_y, y = y
            if isinstance(mask_pred, torch.Tensor):
                mask_pred, pred = mask_pred.cpu(), pred.cpu()
                mask_y, y = mask_y.cpu(), y.cpu()
            n_rows = 4
        else:
            if isinstance(y, torch.Tensor):
                y, pred = y.cpu(), pred.cpu()
    else:
        n_rows = n_rows + len(pred) - 1
        noise_sources_target, y = y
        *noise_sources_pred, y_pred = pred
        y_pred = y_pred.cpu() if isinstance(y_pred, torch.Tensor) else y_pred
        pred = [el.unsqueeze(-1).cpu() if isinstance(el, torch.Tensor) else el for el in noise_sources_pred] + [y_pred]
        y = y.cpu() if isinstance(y, torch.Tensor) else y

    if isinstance(x, torch.Tensor):
        x = x.cpu()

    img_wid = 8 + 2 if separate_components else 8
    figure, axes = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=(2 * time_steps, img_wid))
    min_x, max_x = x[b_idx].min(), x[b_idx].max()
    # min_pred, max_pred = pred[b_idx].min(), pred[b_idx].max()
    min_y, max_y = y[b_idx].min(), y[b_idx].max()
    if min_y == max_y:
        min_y -= 1.
        max_y += 1.
        # min_pred -= 1.
        # max_pred += 1.

    '''if abs(min_y) == 0.:
        min_y = None
        max_y = None
        min_pred = None
        max_pred = None'''

    if n_rows == 3:
        data_list = [x[b_idx], pred[b_idx], y[b_idx]]
        min_list = [min_x, min_y, min_y]  # [min_x, min_pred, min_y]  # use pred scale to compare retrieved amplitude
        max_list = [max_x, max_y, max_y]  # [max_x, max_pred, max_y]
        ax_titles = ['data', 'pred', 'target']
    elif n_rows == 4:
        data_list = [x[b_idx], mask_pred[b_idx], pred[b_idx], y[b_idx]]
        min_list = [min_x, 0., min_y, min_y]
        max_list = [max_x, 1., max_y, max_y]
        ax_titles = ['data', 'mask', 'disp', 'target']
    else:
        # separate components
        pred = [p[b_idx] for p in pred]
        data_list = [x[b_idx]] + [p for p in pred] + [y[b_idx]]
        min_list = [min_x] + [p.min() for p in pred] + [min_y] #[min_x, 0., min_y, min_y]
        max_list = [max_x] + [p.max() for p in pred] + [max_y]# [max_x, 1., max_y, max_y]
        ax_titles = ['data', 'stratified_linear', 'stratified_quadratic', 'turbulent', 'transient', 'mogi', 'fault', 'target']

    for i in range(n_rows):
        if min_list[i] < 0. < max_list[i]:
            norm = TwoSlopeNorm(vmin=min_list[i], vcenter=0., vmax=max_list[i])
            cmap = plt.get_cmap('RdBu_r')
        else:
            norm = Normalize(vmin=min_list[i], vmax=max_list[i])
            cmap = plt.get_cmap('Blues')
        data = data_list[i]
        for j in range(n_cols):
            im = axes[i, j].imshow(data[j, ..., 0], norm=norm, cmap=cmap)
            axes[i, j].get_xaxis().set_ticks([])
            axes[i, j].get_yaxis().set_ticks([])
            if not separate_components:
                if j == 4:
                    axes[i, j].set_title(ax_titles[i])
            else:
                if j == 0:
                    axes[i, j].set_title(ax_titles[i])
        cbar = plt.colorbar(im, ax=axes[i].ravel().tolist(), shrink=0.1, pad=0., orientation='horizontal')
        # cbar.set_label('[mm]', rotation=90, labelpad=10)
    # plt.tight_layout()
    return figure


def assemble_image_plots(data_tuple, pred, batch_idx):
    data, y = data_tuple
    x = data[0]  # exclude topography

    if isinstance(batch_idx, list) or isinstance(batch_idx, tuple):
        figures = []
        for idx in batch_idx:
            # x, y, pred = x.cpu(), y.cpu(), pred.cpu()
            figure = create_fig_time_series(x, pred, y, idx)
            figures.append(figure)
        return figures
    else:
        figure = create_fig_time_series(x, pred, y, batch_idx)
        return figure
