from typing import List, Optional, Union, Tuple

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

from kito.callbacks.tensorboard_callback_images import BaseImagePlotter


class InSARTimeSeriesPlotter(BaseImagePlotter):
    """
    Plots InSAR time series data to TensorBoard.

    Handles multiple output scenarios:
    - Simple prediction (3 rows: data, pred, target)
    - Multi-task with mask (4 rows: data, mask, disp, target)
    - Separate components (multiple rows for different noise sources)

    Args:
        log_dir: Directory for TensorBoard logs
        tag: Tag for images in TensorBoard (default: 'insar_timeseries')
        freq: Plot every N epochs (default: 1)
        batch_indices: Which batch samples to visualize (default: [0])

    Example:
        plotter = InSARTimeSeriesPlotter(
            log_dir='logs/tensorboard',
            tag='predictions',
            freq=5,
            batch_indices=[0, 1, 2]
        )

        engine.fit(train_loader, val_loader, callbacks=[plotter])
    """

    def __init__(
            self,
            log_dir: Optional[str] = None,
            tag: str = 'insar_timeseries',
            freq: int = 1,
            batch_indices: Optional[List[int]] = None
    ):
        super().__init__(log_dir, tag, freq, batch_indices)

    def create_figure(
            self,
            val_data: Tuple,
            val_outputs: Union[torch.Tensor, List, Tuple],
            epoch: int
    ):
        """
        Create matplotlib figures for InSAR time series visualization.

        Args:
            val_data: Tuple of (inputs, targets)
            val_outputs: Model predictions (can be tensor, list, or tuple)
            epoch: Current epoch number

        Returns:
            Single figure or list of figures (one per batch index)
        """
        # Unpack validation data
        inputs, targets = val_data

        # Extract x (data) - exclude topography if present
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]  # First element is data, rest might be topography
        else:
            x = inputs

        y = targets
        pred = val_outputs

        # validate batch indices against actual batch size
        batch_size = x.shape[0]
        valid_indices = [idx for idx in self.batch_indices if idx < batch_size]

        # Warn if some indices are out of bounds
        if len(valid_indices) < len(self.batch_indices):
            invalid_indices = [idx for idx in self.batch_indices if idx >= batch_size]
            import warnings
            warnings.warn(
                f"InSARTimeSeriesPlotter: Batch indices {invalid_indices} exceed "
                f"batch size {batch_size}. These will be skipped. "
                f"Valid indices: {valid_indices}"
            )

        # Handle case where all indices are invalid
        if len(valid_indices) == 0:
            import warnings
            warnings.warn(
                f"InSARTimeSeriesPlotter: All batch indices {self.batch_indices} "
                f"exceed batch size {batch_size}. Using index [0] as fallback."
            )
            valid_indices = [0]

        # Use validated indices
        batch_indices_to_plot = valid_indices

        # Create figures for each batch index
        if len(batch_indices_to_plot) > 1:
            figures = []
            for b_idx in batch_indices_to_plot:
                fig = self._create_single_figure(x, pred, y, b_idx)
                figures.append(fig)
            return figures
        else:
            return self._create_single_figure(x, pred, y, batch_indices_to_plot[0])

    def _create_single_figure(
            self,
            x: torch.Tensor,
            pred: Union[torch.Tensor, List, Tuple],
            y: Union[torch.Tensor, List, Tuple],
            b_idx: int
    ):
        """
        Create a single figure for one batch sample.

        Args:
            x: Input data [B, T, H, W, C]
            pred: Predictions (tensor or list/tuple for multi-output)
            y: Target (tensor or list/tuple for multi-output)
            b_idx: Batch index to visualize

        Returns:
            matplotlib.figure.Figure
        """
        # Determine if we have separate components (multi-output scenario)
        separate_components = (
                isinstance(pred, (list, tuple)) and len(pred) > 2
        )

        time_steps = x.shape[1]
        n_rows, n_cols = 3, time_steps

        # ===== Scenario 1: Separate components (multiple noise sources) =====
        if separate_components:
            n_rows = 3 + len(pred) - 1
            noise_sources_target, y = y
            *noise_sources_pred, y_pred = pred

            # Move to CPU
            y_pred = y_pred.cpu() if isinstance(y_pred, torch.Tensor) else y_pred
            pred = [
                       el.unsqueeze(-1).cpu() if isinstance(el, torch.Tensor) else el
                       for el in noise_sources_pred
                   ] + [y_pred]
            y = y.cpu() if isinstance(y, torch.Tensor) else y

            ax_titles = [
                'data', 'stratified_linear', 'stratified_quadratic',
                'turbulent', 'transient', 'mogi', 'fault', 'target'
            ]

        # ===== Scenario 2: Multi-task with mask =====
        elif isinstance(pred, (list, tuple)) and len(pred) == 2:
            mask_pred, pred = pred
            mask_y, y = y

            if isinstance(mask_pred, torch.Tensor):
                mask_pred, pred = mask_pred.cpu(), pred.cpu()
                mask_y, y = mask_y.cpu(), y.cpu()

            n_rows = 4
            ax_titles = ['data', 'mask', 'disp', 'target']

        # ===== Scenario 3: Simple prediction =====
        else:
            if isinstance(y, torch.Tensor):
                y, pred = y.cpu(), pred.cpu()

            ax_titles = ['data', 'pred', 'target']

        # Move x to CPU
        if isinstance(x, torch.Tensor):
            x = x.cpu()

        # ===== Create figure =====
        img_wid = 8 + 2 if separate_components else 8
        figure, axes = plt.subplots(
            n_rows, n_cols,
            sharex='all', sharey='all',
            figsize=(2 * time_steps, img_wid)
        )

        # ===== Compute normalization ranges =====
        min_x, max_x = x[b_idx].min(), x[b_idx].max()
        min_y, max_y = y[b_idx].min(), y[b_idx].max()

        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0

        # ===== Prepare data list based on scenario =====
        if n_rows == 3:
            data_list = [x[b_idx], pred[b_idx], y[b_idx]]
            min_list = [min_x, min_y, min_y]
            max_list = [max_x, max_y, max_y]

        elif n_rows == 4:
            data_list = [x[b_idx], mask_pred[b_idx], pred[b_idx], y[b_idx]]
            min_list = [min_x, 0.0, min_y, min_y]
            max_list = [max_x, 1.0, max_y, max_y]

        else:  # separate components
            pred = [p[b_idx] for p in pred]
            data_list = [x[b_idx]] + pred + [y[b_idx]]
            min_list = [min_x] + [p.min() for p in pred] + [min_y]
            max_list = [max_x] + [p.max() for p in pred] + [max_y]

        # ===== Plot each row =====
        for i in range(n_rows):
            # Choose colormap and normalization
            if min_list[i] < 0.0 < max_list[i]:
                norm = TwoSlopeNorm(
                    vmin=min_list[i],
                    vcenter=0.0,
                    vmax=max_list[i]
                )
                cmap = plt.get_cmap('RdBu_r')
            else:
                norm = Normalize(vmin=min_list[i], vmax=max_list[i])
                cmap = plt.get_cmap('Blues')

            data = data_list[i]

            # Plot each time step
            for j in range(n_cols):
                im = axes[i, j].imshow(data[j, ..., 0], norm=norm, cmap=cmap)
                axes[i, j].get_xaxis().set_ticks([])
                axes[i, j].get_yaxis().set_ticks([])

                # Add titles
                if not separate_components:
                    if j == n_cols // 2:  # Middle column
                        axes[i, j].set_title(ax_titles[i])
                else:
                    if j == 0:  # First column
                        axes[i, j].set_title(ax_titles[i])

            # Add colorbar for row
            cbar = plt.colorbar(
                im,
                ax=axes[i].ravel().tolist(),
                shrink=0.1,
                pad=0.0,
                orientation='horizontal'
            )

        return figure
