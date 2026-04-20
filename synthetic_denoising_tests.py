import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
#from cmcrameri import cm as scmap
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from config import datagen_params
from config.local_denoiser_params import denoiser_configuration
from datagenerator.DataGeneratorV2 import DataGeneratorV2
from datagenerator.PureNoiseGeneratorV2 import PureNoiseGeneratorV2
from dataset.insarmemdataset import InSARMemDataset
from ismetpasa_denoising import compute_displacement_from_extrema, profile_along_segment
from kito import GenericDataPipeline, Engine

from model.insar_denoiser import InSARDenoiser
from utils.metrics_utils import signal_to_noise_ratio_2d
from utils.stats_utils import gaussian_cdf

SMALL_SIZE = 14 + 4
LEGEND_SIZE = 14 + 4
MEDIUM_SIZE = 16 + 4
LARGE_SIZE = 20 + 4
FONT_LIST = ['Helvetica', 'DejaVu Sans']

style_dict = {
    'font.size': SMALL_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_LIST,  # Helvetica if available, else DejaVu
    'axes.titlesize': LARGE_SIZE,
    'axes.labelsize': SMALL_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.titlesize': LARGE_SIZE
}

plt.rcParams.update(style_dict)


def init_params():
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}
    params['batch_size'] = 1
    params['rand_noise_sigma'] = 0.
    params['decorrelation_pixel_value_range'] = (-1e-20, 1e-20)
    params['glitchy_pixel_amplitude_max'] = 1e-20
    params['buggy_patches_frac_max'] = 0.
    params['glitch_frac_exp_lim'] = (-20, -19)
    return params


def complex_fault():
    # Image dimensions
    nx, ny = 128, 128

    # Define fault trace with kinks
    fault_points = np.array([
        [0, 50],  # Start
        [50, 60],  # First kink
        [90, 45],  # Second kink
        [127, 55]  # End
    ])

    # Interpolate fault trace
    x_trace = np.linspace(fault_points[0, 0], fault_points[-1, 0], 500)
    y_trace = np.interp(x_trace, fault_points[:, 0], fault_points[:, 1])

    # Create pixel grid
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    pixels = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Fault points
    fault_xy = np.stack([x_trace, y_trace], axis=1)

    # Compute signed distance to fault for each pixel
    signed_distances = []

    for px, py in pixels:
        # Find nearest fault point
        dists = np.sqrt((fault_xy[:, 0] - px) ** 2 + (fault_xy[:, 1] - py) ** 2)
        idx_nearest = np.argmin(dists)
        min_dist = dists[idx_nearest]

        # Sign: positive above, negative below
        sign = 1.0 if py < y_trace[idx_nearest] else -1.0
        # sign = -1.0 if py < y_trace[idx_nearest] else 1.0
        signed_distances.append(sign * min_dist)

    signed_distances = np.array(signed_distances).reshape(ny, nx)

    # COSEISMIC DISPLACEMENT MODEL
    # Maximum at fault, decays exponentially with distance
    D_max = 5.0  # Maximum slip at fault (cm)
    decay_width = 25.0  # Decay length scale (pixels)

    # Exponential decay with sign
    displacement = np.sign(signed_distances) * D_max * np.exp(-np.abs(signed_distances) / decay_width)

    # Plot
    '''fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Displacement field
    im = axes[0].imshow(displacement, cmap='RdBu_r', vmin=-D_max, vmax=D_max)
    axes[0].plot(x_trace, y_trace, 'k-', linewidth=2.5, label='Fault trace')
    axes[0].scatter(fault_points[:, 0], fault_points[:, 1], c='yellow', s=50,
                    zorder=5, edgecolors='black', linewidth=1.5, label='Key points')
    axes[0].set_title('Coseismic strike-slip displacement')
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    axes[0].legend()
    plt.colorbar(im, ax=axes[0], label='Displacement (cm)')

    # Cross-section
    middle_x = nx // 2
    fault_y = y_trace[np.argmin(np.abs(x_trace - middle_x))]

    axes[1].plot(displacement[:, middle_x], np.arange(ny), 'b-', linewidth=2.5, label='Displacement')
    axes[1].axhline(y=fault_y, color='red', linestyle='--', linewidth=2, label='Fault position')
    axes[1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Displacement (cm)')
    axes[1].set_ylabel('y (pixels)')
    axes[1].set_title(f'Cross-section at x={middle_x}')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()'''

    # print(f"Max displacement at fault: ±{D_max:.1f} cm")
    # print(f"Displacement at {decay_width:.0f} pixels from fault: ±{D_max * np.exp(-1):.1f} cm")
    tspace = np.arange(9)
    gaussian_rate = np.array([gaussian_cdf(val, 3, 2.) for val in tspace])
    fault_deformation = (gaussian_rate[:, None, None] * displacement)[..., None]
    return fault_deformation


def generate_noise_ts(params, seed):
    generator = PureNoiseGeneratorV2(params, dtype=float, verbose=False, seed=seed)
    print('using seed:', generator.seed)
    batch_x, batch_y, batch_topo = generator.generate_batch_data(return_components=False)
    return batch_x, batch_topo


def slow_rupture(params, seed):
    params['n_faults'] = 1
    params['fault_proba'] = 1.
    params['mogi_proba'] = 0.
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['depth_range'] = (0, 0.1)
    params['fault_length_range'] = (100, 101)  # km
    params['fault_rake_range'] = (178, 180)  # degrees
    params['fault_dip_range'] = (88, 89)  # degrees
    params['lat_lon_moving_ratio_inside_window'] = .95
    params['fault_slip_range'] = (40., 40.1)  # cm
    # params['bilateral_noise_amplitude_range'] = (-1e-20, 1e-20)
    generator = DataGeneratorV2(params, dtype=float, verbose=False, seed=seed)
    print('using seed:', generator.seed)
    batch_x, batch_y, batch_topo = generator.generate_batch_data(return_components=False)
    return batch_x, batch_y, batch_topo


def fast_rupture(params, seed):
    params['n_faults'] = 1
    params['fault_proba'] = 1.
    params['mogi_proba'] = 0.
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['depth_range'] = (5., 5.1)
    params['fault_length_range'] = (300, 301)  # km
    params['fault_rake_range'] = (90, 91)  # degrees
    params['fault_dip_range'] = (75, 76)  # degrees
    params['lat_lon_moving_ratio_inside_window'] = .95
    params['fault_slip_range'] = (50., 50.1)  # cm
    params['fault_slip_onset_time_range'] = (3, 4)
    params['fault_slip_duration_sigma_range'] = (.00001, .000011)
    params['fault_slip_duration_sigma_min_inside_win'] = 0.
    # params['bilateral_noise_amplitude_range'] = (-1e-20, 1e-20)
    generator = DataGeneratorV2(params, dtype=float, verbose=False, seed=seed)
    print('using seed:', generator.seed)
    batch_x, batch_y, batch_topo = generator.generate_batch_data(return_components=False)
    return batch_x, batch_y, batch_topo


def predict(denoiser_configuration, x, y, topo):
    denoiser_configuration.data.total_samples = len(x)
    denoiser_configuration.training.batch_size = 32
    denoiser_configuration.data.train_ratio = 0.
    denoiser_configuration.data.val_ratio = 0.

    dataset = InSARMemDataset(x=x, y=y, topo=topo)

    pipeline = GenericDataPipeline(
        config=denoiser_configuration,
        dataset=dataset,
        preprocessing=denoiser_configuration.data.preprocessing,
    )

    pipeline.setup()

    denoiser_configuration.training.train_mode = False
    denoiser_configuration.model.save_inference_to_disk = False

    denoiser = InSARDenoiser(denoiser_configuration)
    engine = Engine(denoiser, denoiser_configuration)

    engine.module.build()
    engine.load_weights(denoiser_configuration.model.weight_load_path)

    prediction = engine.predict(data_pipeline=pipeline)

    return prediction


def plot_results(data_examples, truth_examples, pred_examples):
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(13, 9, figure=fig,
                  height_ratios=[0.3, 1, 1, 1, 0.3, 1, 1, 1, 0.3, 1, 1, 1, 0.1],
                  hspace=0.3, wspace=0.2,
                  left=0.05, right=0.88, top=0.95, bottom=0.05)

    row_labels = ['Input', 'Ground truth', 'Prediction']

    # ---------- helper for one subgroup ----------
    def plot_subgroup(example_idx, row_start, title, data_norm, truth_norm):
        """
        example_idx: 0,1,2 selects data_examples[example_idx], truth_examples[example_idx], pred_examples[example_idx]
        row_start: top data row index in GridSpec (1,5,9)
        title: string
        data_norm, truth_norm: TwoSlopeNorm for noisy vs GT/Pred
        """
        cmap = 'RdBu_r'  # scmap.vik
        # Title spanning middle columns
        ax_title = fig.add_subplot(gs[row_start - 1, 3:6])
        ax_title.text(0.5, 0.5, title, ha='center', va='center', font='DejaVu Sans',
                      fontsize=16, weight='bold', transform=ax_title.transAxes)
        ax_title.axis('off')

        # Store one axis per row (first column) and one axis at right for colorbar placement
        row_axes = {}
        right_axes = {}

        # Plot 3×9 block
        for row_idx in range(3):  # 0: noisy, 1: GT, 2: pred
            for t in range(9):
                ax = fig.add_subplot(gs[row_start + row_idx, t])

                if row_idx == 0:
                    img = data_examples[example_idx, t]
                    im_data = ax.imshow(img, cmap=cmap, norm=data_norm, origin='lower')
                elif row_idx == 1:
                    img = truth_examples[example_idx, t]
                    im_truth = ax.imshow(img, cmap=cmap, norm=truth_norm, origin='lower')
                else:
                    img = pred_examples[example_idx, t]
                    ax.imshow(img, cmap=cmap, norm=truth_norm, origin='lower')

                ax.set_xticks([])
                ax.set_yticks([])

                if t == 0:
                    ax.set_ylabel(row_labels[row_idx], font='DejaVu Sans', fontsize=12, weight='bold', labelpad=5)
                    row_axes[row_idx] = ax  # reference for this row

                if t == 8:
                    right_axes[row_idx] = ax  # last column for x-position

                if row_idx == 0 and example_idx == 0:
                    ax.set_title(f't={t + 1}', fontsize=14)

        # Ensure layout is computed
        fig.canvas.draw()

        # Use row 0 left & right axes to get x and height for the top colorbar
        ax0_left = row_axes[0]
        ax0_right = right_axes[0]

        b0 = ax0_left.get_position()
        b0r = ax0_right.get_position()

        x_cbar = b0r.x1 + 0.01
        w_cbar = 0.015 / 2.

        # Row 0 (data) colorbar: height = one row
        y0_data = b0.y0
        h_data = b0.height

        cax_data = fig.add_axes([x_cbar, y0_data, w_cbar, h_data])
        cbar_data = fig.colorbar(im_data, cax=cax_data)
        cbar_data.set_label('LOS disp. [cm]', fontsize=15, weight='bold', labelpad=10)

        # Rows 1–2 (GT + Pred) colorbar: span those two rows
        ax1_left = row_axes[1]
        ax2_left = row_axes[2]
        b1 = ax1_left.get_position()
        b2 = ax2_left.get_position()

        bottom = min(b1.y0, b2.y0)
        top = max(b1.y1, b2.y1)
        h_gtp = top - bottom

        cax_gtp = fig.add_axes([x_cbar, bottom, w_cbar, h_gtp])
        cbar_gtp = fig.colorbar(im_truth, cax=cax_gtp)
        if example_idx == 1:
            # force the lower bound...
            cbar_gtp.ax.set_yscale('linear')
            cbar_gtp.set_ticks([-3, 0, 5, 10])
            cbar_gtp.set_ticklabels(['-3', '0', '5', '10'])
            # cbar_gtp.set_ticks([-3, -2, -1, 0, 5, 10])
            # cbar_gtp.set_ticklabels(['-3', '-2', '-1', '0', '5', '10'])
        else:
            cbar_gtp.ax.set_yscale('linear')


        cbar_gtp.set_label('LOS displacement [cm]', fontsize=15, weight='bold', labelpad=10)

    snr_list = signal_to_noise_ratio_2d(data_examples - truth_examples, truth_examples)

    # ---------- subgroup 1: rows 1–3 ----------
    norm1_data = TwoSlopeNorm(vmin=data_examples[0].min(), vcenter=0, vmax=data_examples[0].max())
    norm1_truth = TwoSlopeNorm(vmin=truth_examples[0].min(), vcenter=0, vmax=truth_examples[0].max())
    plot_subgroup(example_idx=0, row_start=1, title=rf'$\mathbf{{Slow\ rupture\ (SNR={snr_list[0]:.2f})}}$',
                  data_norm=norm1_data, truth_norm=norm1_truth)

    # ---------- subgroup 2: rows 5–7 ----------
    norm2_data = TwoSlopeNorm(vmin=data_examples[1].min(), vcenter=0, vmax=data_examples[1].max())
    norm2_truth = TwoSlopeNorm(vmin=truth_examples[1].min(), vcenter=0, vmax=truth_examples[1].max())
    plot_subgroup(example_idx=1, row_start=5, title=rf'$\mathbf{{Fast\ rupture\ (SNR={snr_list[1]:.2f})}}$',
                  data_norm=norm2_data, truth_norm=norm2_truth)

    # ---------- subgroup 3: rows 9–11 ----------
    norm3_data = TwoSlopeNorm(vmin=data_examples[2].min(), vcenter=0, vmax=data_examples[2].max())
    norm3_truth = TwoSlopeNorm(vmin=truth_examples[2].min(), vcenter=0, vmax=truth_examples[2].max())
    plot_subgroup(example_idx=2, row_start=9, title=rf'$\mathbf{{Complex\ fault\ trace\ (SNR={snr_list[2]:.2f})}}$',
                  data_norm=norm3_data, truth_norm=norm3_truth)
    # plt.show()
    plt.savefig('/Users/giuseppe/Documents/DATA/InSAR/insar-denoising/figures/synthetic_denoising_test.pdf',
                bbox_inches='tight')


def plot_results_v2(data_examples, truth_examples, pred_examples):
    residual_ex = np.abs(truth_examples - pred_examples)

    fig = plt.figure(figsize=(20, 16+5))
    gs = GridSpec(13+3, 9, figure=fig,
                  height_ratios=[0.3, 1, 1, 1, 1, 0.3, 1, 1, 1, 1, 0.3, 1, 1, 1, 1, 0.1],
                  hspace=0.3, wspace=0.2,
                  left=0.05, right=0.88, top=0.95, bottom=0.05)

    row_labels = ['Input', 'Ground truth', 'Prediction', 'Residuals']

    # ---------- helper for one subgroup ----------
    def plot_subgroup(example_idx, row_start, title, data_norm, truth_norm, res_norm):
        """
        example_idx: 0,1,2 selects data_examples[example_idx], truth_examples[example_idx], pred_examples[example_idx]
        row_start: top data row index in GridSpec (1,5,9)
        title: string
        data_norm, truth_norm: TwoSlopeNorm for noisy vs GT/Pred
        """
        cmap = 'RdBu_r'  # scmap.vik
        # Title spanning middle columns
        ax_title = fig.add_subplot(gs[row_start - 1, 3:6])
        ax_title.text(0.5, 0.5, title, ha='center', va='center', font='DejaVu Sans',
                      fontsize=16, weight='bold', transform=ax_title.transAxes)
        ax_title.axis('off')

        # Store one axis per row (first column) and one axis at right for colorbar placement
        row_axes = {}
        right_axes = {}

        # Plot 3×9 block
        for row_idx in range(4):  # 0: noisy, 1: GT, 2: pred
            for t in range(9):
                ax = fig.add_subplot(gs[row_start + row_idx, t])

                if row_idx == 0:
                    img = data_examples[example_idx, t]
                    im_data = ax.imshow(img, cmap=cmap, norm=data_norm, origin='lower')
                elif row_idx == 1:
                    img = truth_examples[example_idx, t]
                    im_truth = ax.imshow(img, cmap=cmap, norm=truth_norm, origin='lower')
                elif row_idx == 2:
                    img = pred_examples[example_idx, t]
                    ax.imshow(img, cmap=cmap, norm=truth_norm, origin='lower')
                else:
                    img = residual_ex[example_idx, t]
                    im_res = ax.imshow(img, cmap='Blues', norm=res_norm, origin='lower')

                ax.set_xticks([])
                ax.set_yticks([])

                if t == 0:
                    ax.set_ylabel(row_labels[row_idx], font='DejaVu Sans', fontsize=12, weight='bold', labelpad=5)
                    row_axes[row_idx] = ax  # reference for this row

                if t == 8:
                    right_axes[row_idx] = ax  # last column for x-position

                if row_idx == 0 and example_idx == 0:
                    ax.set_title(f't={t + 1}', fontsize=14)

        # Ensure layout is computed
        fig.canvas.draw()

        # Use row 0 left & right axes to get x and height for the top colorbar
        ax0_left = row_axes[0]
        ax0_right = right_axes[0]

        b0 = ax0_left.get_position()
        b0r = ax0_right.get_position()

        x_cbar = b0r.x1 + 0.01
        w_cbar = 0.015 / 2.

        # Row 0 (data) colorbar: height = one row
        y0_data = b0.y0
        h_data = b0.height

        cax_data = fig.add_axes([x_cbar, y0_data, w_cbar, h_data])
        cbar_data = fig.colorbar(im_data, cax=cax_data)
        cbar_data.set_label('LOS disp. [cm]', fontsize=15, weight='bold', labelpad=10)

        # Rows 1–2 (GT + Pred) colorbar: span those two rows
        ax1_left = row_axes[1]
        ax2_left = row_axes[2]
        b1 = ax1_left.get_position()
        b2 = ax2_left.get_position()

        bottom = min(b1.y0, b2.y0)
        top = max(b1.y1, b2.y1)
        h_gtp = top - bottom

        cax_gtp = fig.add_axes([x_cbar, bottom, w_cbar, h_gtp])
        cbar_gtp = fig.colorbar(im_truth, cax=cax_gtp)
        if example_idx == 1:
            # force the lower bound...
            cbar_gtp.ax.set_yscale('linear')
            cbar_gtp.set_ticks([-3, 0, 5, 10])
            cbar_gtp.set_ticklabels(['-3', '0', '5', '10'])
            # cbar_gtp.set_ticks([-3, -2, -1, 0, 5, 10])
            # cbar_gtp.set_ticklabels(['-3', '-2', '-1', '0', '5', '10'])
        else:
            cbar_gtp.ax.set_yscale('linear')

        # residuals colorbar
        ax3_left = row_axes[3]
        b3 = ax3_left.get_position()
        y3_data = b3.y0
        cax_data = fig.add_axes([x_cbar, y3_data, w_cbar, h_data])
        cbar_data = fig.colorbar(im_res, cax=cax_data)
        cbar_data.set_label('LOS disp. [cm]', fontsize=15, weight='bold', labelpad=10)

        cbar_gtp.set_label('LOS displacement [cm]', fontsize=15, weight='bold', labelpad=10)

    snr_list = signal_to_noise_ratio_2d(data_examples - truth_examples, truth_examples)

    # ---------- subgroup 1: rows 1–3 ----------
    norm1_data = TwoSlopeNorm(vmin=data_examples[0].min(), vcenter=0, vmax=data_examples[0].max())
    norm1_truth = TwoSlopeNorm(vmin=truth_examples[0].min(), vcenter=0, vmax=truth_examples[0].max())
    # norm1_res = TwoSlopeNorm(vmin=residual_ex[0].min(), vcenter=0, vmax=residual_ex[0].max())
    norm1_res = Normalize(vmin=residual_ex[0].min(), vmax=residual_ex[0].max())
    plot_subgroup(example_idx=0, row_start=1, title=rf'$\mathbf{{Slow\ rupture\ (SNR={snr_list[0]:.2f})}}$',
                  data_norm=norm1_data, truth_norm=norm1_truth, res_norm=norm1_res)

    # ---------- subgroup 2: rows 5–7 ----------
    norm2_data = TwoSlopeNorm(vmin=data_examples[1].min(), vcenter=0, vmax=data_examples[1].max())
    norm2_truth = TwoSlopeNorm(vmin=truth_examples[1].min(), vcenter=0, vmax=truth_examples[1].max())
    # norm2_res = TwoSlopeNorm(vmin=residual_ex[1].min(), vcenter=0, vmax=residual_ex[1].max())
    norm2_res = Normalize(vmin=residual_ex[1].min(), vmax=residual_ex[1].max())
    plot_subgroup(example_idx=1, row_start=5+1, title=rf'$\mathbf{{Fast\ rupture\ (SNR={snr_list[1]:.2f})}}$',
                  data_norm=norm2_data, truth_norm=norm2_truth, res_norm=norm2_res)

    # ---------- subgroup 3: rows 9–11 ----------
    norm3_data = TwoSlopeNorm(vmin=data_examples[2].min(), vcenter=0, vmax=data_examples[2].max())
    norm3_truth = TwoSlopeNorm(vmin=truth_examples[2].min(), vcenter=0, vmax=truth_examples[2].max())
    # norm3_res = TwoSlopeNorm(vmin=residual_ex[2].min(), vcenter=0, vmax=residual_ex[2].max())
    norm3_res = Normalize(vmin=residual_ex[2].min(), vmax=residual_ex[2].max())
    plot_subgroup(example_idx=2, row_start=9+2, title=rf'$\mathbf{{Complex\ fault\ trace\ (SNR={snr_list[2]:.2f})}}$',
                  data_norm=norm3_data, truth_norm=norm3_truth, res_norm=norm3_res)

    #plt.show()
    plt.savefig('/Users/giuseppe/Documents/DATA/InSAR/insar-denoising/figures/supp/synthetic_denoising_test_res.pdf',
                bbox_inches='tight')



def normalize_topo(topo):
    min_topo, max_topo = np.min(topo, axis=(2, 3), keepdims=True), np.max(topo, axis=(2, 3), keepdims=True)
    norm_topo = (topo - min_topo) / (max_topo - min_topo)
    return norm_topo

def plot_transects_synth_post(x, y, pred, segment, ev_type, cbar_pos='bottom'):
    # segment = [[38, 93], [100, 60]]
    # segment = [[38, 93-10], [100, 60-10]]


    img_pred = pred[-1] - pred[0]
    img_data = x[-1] - x[0]
    img_gt = y[-1] - y[0]


    dist_pred, profile_pred = profile_along_segment(img_pred, segment[0], segment[1], n_samples=200, pixel_size=1.0)
    dist_data, profile_data = profile_along_segment(img_data, segment[0], segment[1], n_samples=200, pixel_size=1.0)
    dist_gt, profile_gt = profile_along_segment(img_gt, segment[0], segment[1], n_samples=200, pixel_size=1.0)

    fig = plt.figure(figsize=(14, 7))

    # Create outer gridspec: 1 row, 2 columns
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Left column: single subplot
    ax_left = fig.add_subplot(outer[0, 0])

    # Right column: nested gridspec with 2 rows
    inner_right = outer[0, 1].subgridspec(2, 1, hspace=.4, wspace=0.)

    # Create the two subplots in the right column
    ax_right_top = fig.add_subplot(inner_right[0, 0])
    ax_right_bottom = fig.add_subplot(inner_right[1, 0])

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'wspace': 0.35})

    # -------------------------------------------------
    # LEFT: image with transects
    # -------------------------------------------------
    norm = TwoSlopeNorm(vmin=img_pred.min(), vcenter=0, vmax=img_pred.max())

    im = ax_left.imshow(
        img_pred,
        cmap='RdBu_r',
        norm=norm,
        origin='lower'
    )

    for (p0, p1) in [segment]:
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        ax_left.plot(xs, ys, color='k', lw=2)

        # Annotate with offset
        ax_left.annotate('A', xy=(p0[0], p0[1]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=14, font='DejaVu Sans', fontweight='bold', color='k')

        ax_left.annotate('A\'', xy=(p1[0], p1[1]),
                         xytext=(5, -5), textcoords='offset points',
                         fontsize=14, font='DejaVu Sans', fontweight='999', color='k')

    ax_left.set_aspect('equal')
    # ax_img.set_title('Cumulative deformation')
    ax_left.set_xlabel('Longitude (pixels)', font='DejaVu Sans', fontweight='bold')
    ax_left.set_ylabel('Latitude (pixels)', font='DejaVu Sans', fontweight='bold')

    fig.canvas.draw()  # make sure positions are known

    acc_d, acc_d_std = [], []
    accy_d, accy_d_std = [], []

    for i in range(9):
        dd_i, pp_i = profile_along_segment(pred[i], segment[0], segment[1], n_samples=200, pixel_size=1.0)  # 128/2
        ddy_i, ppy_i = profile_along_segment(y[i], segment[0], segment[1], n_samples=200, pixel_size=1.0)  # 128/2
        # dd_i, pp_i = dd_i[13:], pp_i[13:]
        if np.sum(np.isnan(pp_i)) > 0:
            nan_mask = np.isnan(pp_i)
            not_nan_mask = ~nan_mask
            pp_i = np.interp(dd_i, dd_i[not_nan_mask], pp_i[not_nan_mask])

        delta_px = 30  # 10
        delta_px_trans = 5  # 10
        smoothing_sigma = 3.
        res = compute_displacement_from_extrema(dd_i, pp_i, delta_x=(delta_px, delta_px_trans),
                                                smooth_sigma=smoothing_sigma, median=False,
                                                min_idx=None, max_idx=None, bilateral=False, method='median')

        res_y = compute_displacement_from_extrema(ddy_i, ppy_i, delta_x=(delta_px, delta_px_trans),
                                                smooth_sigma=smoothing_sigma, median=False,
                                                min_idx=None, max_idx=None, bilateral=False, method='median')

        # acc_d.append(np.abs(pp_i[np.where((dd_i - 60)< 0.01)[0][0]] - pp_i[np.where((dd_i - 75)< 0.01)[0][0]]))
        acc_d.append(res['displacement'])
        acc_d_std.append(res['uncertainty'])
        accy_d.append(res_y['displacement'])
        accy_d_std.append(res_y['uncertainty'])

    bbox = ax_left.get_position()  # Bbox in figure coords [web:278]

    # Example: bar spanning 60% of the axes width, 6% of its height,
    # positioned slightly above the bottom-left corner
    w = 0.2 * bbox.width
    h = 0.03 * bbox.height

    if cbar_pos == 'bottom':
        x0 = bbox.x0 + 0.2 * bbox.width
        y0 = bbox.y0 + 0.2 * bbox.height

    if cbar_pos == 'top':
        x0 = bbox.x0 + 0.7 * bbox.width
        y0 = bbox.y0 + 0.9 * bbox.height

    cax = fig.add_axes([x0, y0, w, h])  # explicit coordinates [web:271][web:272]

    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('LOS disp. [cm]')
    cax.xaxis.set_ticks_position('bottom')
    cbar.set_ticks([norm.vmin, norm.vcenter, norm.vmax])
    cbar.set_ticklabels([str(int(norm.vmin)), str(norm.vcenter), str(int(norm.vmax))])

    ax_right_top.plot(dist_pred, profile_pred, color='C3', lw=2.)
    # ax_right_top.plot([], [], color='k', lw=2., label='Data')
    # ax_right_top.scatter(dist_data, profile_data, color='k', s=5)
    ax_right_top.scatter(dist_gt, profile_gt, color='k', s=5)

    ax_right_top.plot([], [], color='C3', lw=2., label='Denoised')
    ax_right_top.plot([], [], color='k', lw=2., label='Ground truth')

    ax_right_bottom.plot([], [], color='C3', lw=2., label='Denoised')
    ax_right_bottom.plot([], [], color='k', lw=2., label='Ground truth')


    ax_right_bottom.plot([i for i in range(9)], accy_d,
                             marker='o', color='k', markersize=5)

    ax_right_bottom.errorbar([i for i in range(9)], acc_d, yerr=acc_d_std,
                             fmt='o', color='C3', markersize=5,
                             capsize=3, capthick=1.)



    ax_right_top.set_xlabel('A-A\' distance along transect (pixels)', font='DejaVu Sans',
                            fontweight='bold')  # fontweight='bold', font='DejaVu Sans'
    ax_right_top.set_ylabel('Displacement [cm]', font='DejaVu Sans', fontweight='bold')
    ax_right_top.legend()
    ax_right_bottom.set_xlabel('Synthetic time (acquisition #)', font='DejaVu Sans', fontweight='bold')
    ax_right_bottom.set_ylabel('Displacement [cm]', font='DejaVu Sans', fontweight='bold')
    ax_right_bottom.legend()

    ax_right_bottom.spines[['right', 'top']].set_visible(False)
    ax_right_top.spines[['right', 'top']].set_visible(False)

    # ax_prof.legend(loc='best', fontsize=8)
    # fig.tight_layout()
    # plt.show()
    plt.savefig(f'/Users/giuseppe/Documents/DATA/InSAR/insar-denoising/figures/supp/synth_transects_{ev_type}.pdf',
                bbox_inches='tight')


if __name__ == '__main__':
    params = init_params()
    complex_noise, complex_topo = generate_noise_ts(params, seed=1628961789)  # 1018838343  # 563262684 # 1321548698
    slow_x, slow_y, topo_slow = slow_rupture(params, seed=3263276907)  # 616459241  # 3263276907
    fast_x, fast_y, topo_fast = fast_rupture(params, seed=3495699828)  # 1832672846# 3495699828 # 2570235688 #   # 2161183748
    complex_y = complex_fault()[None, ...]
    complex_y = complex_y - complex_y[:, 0, ...][:, None, ...]  # no knowledge of fault location
    complex_x = complex_y + complex_noise
    # normalize topography
    topo_slow = normalize_topo(topo_slow)
    topo_fast = normalize_topo(topo_fast)
    complex_topo = normalize_topo(complex_topo)

    x_all = np.concatenate((slow_x, fast_x, complex_x), dtype=np.float32)
    y_all = np.concatenate((slow_y, fast_y, complex_y), dtype=np.float32)
    topo_all = np.concatenate((topo_slow, topo_fast, complex_topo), dtype=np.float32)
    pred_all = predict(denoiser_configuration, x_all, y_all, topo_all)

    x_all = x_all.squeeze()
    y_all = y_all.squeeze()
    topo_all = topo_all.squeeze()
    pred_all = pred_all.squeeze()

    plot_results(x_all, y_all, pred_all)
    plot_results_v2(x_all, y_all, pred_all)

    plot_transects_synth_post(x_all[0], y_all[0], pred_all[0], [[40, 80], [40, 10]], 'slow_rupture', cbar_pos='top')
    plot_transects_synth_post(x_all[1], y_all[1], pred_all[1], [[60, 70], [120, 110]], 'fast_rupture')
    plot_transects_synth_post(x_all[2], y_all[2], pred_all[2], [[90, 80], [30, 30]], 'complex_fault_trace', cbar_pos='top')

