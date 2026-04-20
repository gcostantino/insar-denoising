import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec

from datagenerator.DataGeneratorV2 import DataGeneratorV2
from ismetpasa_denoising import profile_along_segment, point_to_pixel_ref, compute_displacement_from_extrema
from synthetic_denoising_tests import init_params, generate_noise_ts, normalize_topo, predict
from utils.metrics_utils import signal_to_noise_ratio_2d

import os

import numpy as np
import scipy.fft as fftw
import scipy.spatial.distance as scidis
from scipy.ndimage import rotate

from okada.okada import forward as okada85
from utils.modeldeform_utils import mogi_defo
from utils.stats_utils import gaussian_cdf

from config.local_denoiser_params import denoiser_configuration

def sample_rupture(params, seed, onset_frame=1, sigma=.5, slip=40.):
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
    params['fault_slip_range'] = (slip, slip + .1)  # cm
    params['fault_slip_onset_time_range'] = (onset_frame, onset_frame + 1)
    params['fault_slip_duration_sigma_range'] = (sigma, sigma + 0.1)
    # params['bilateral_noise_amplitude_range'] = (-1e-20, 1e-20)
    generator = DataGeneratorV2(params, dtype=float, verbose=False, seed=seed)
    print('using seed:', generator.seed)
    batch_x, batch_y, batch_topo = generator.generate_batch_data(return_components=False)
    return batch_x, batch_y, batch_topo


def plot_results_two_ev_residuals(data_examples, truth_examples, pred_examples):
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

        cbar_gtp.ax.set_yscale('linear')

        '''if example_idx == 1:
            # force the lower bound...
            cbar_gtp.set_ticks([-3, 0, 5, 10])
            cbar_gtp.set_ticklabels(['-3', '0', '5', '10'])'''

        # residuals colorbar
        ax3_left = row_axes[3]
        b3 = ax3_left.get_position()
        y3_data = b3.y0
        cax_data = fig.add_axes([x_cbar, y3_data, w_cbar, h_data])
        cbar_data = fig.colorbar(im_res, cax=cax_data)
        cbar_data.set_label('LOS disp. [cm]', fontsize=15, weight='bold', labelpad=10)

        cbar_gtp.set_label('LOS displacement [cm]', fontsize=15, weight='bold', labelpad=10)

    # ---------- subgroup 1: rows 1–3 ----------
    norm1_data = TwoSlopeNorm(vmin=data_examples[0].min(), vcenter=0, vmax=data_examples[0].max())
    norm1_truth = TwoSlopeNorm(vmin=truth_examples[0].min(), vcenter=0, vmax=truth_examples[0].max())
    # norm1_res = TwoSlopeNorm(vmin=residual_ex[0].min(), vcenter=0, vmax=residual_ex[0].max())
    norm1_res = Normalize(vmin=residual_ex[0].min(), vmax=residual_ex[0].max())
    plot_subgroup(example_idx=0, row_start=1, title='Two events test',
                  data_norm=norm1_data, truth_norm=norm1_truth, res_norm=norm1_res)

    # ---------- subgroup 2: rows 5–7 ----------
    '''norm2_data = TwoSlopeNorm(vmin=data_examples[1].min(), vcenter=0, vmax=data_examples[1].max())
    norm2_truth = TwoSlopeNorm(vmin=truth_examples[1].min(), vcenter=0, vmax=truth_examples[1].max())
    # norm2_res = TwoSlopeNorm(vmin=residual_ex[1].min(), vcenter=0, vmax=residual_ex[1].max())
    norm2_res = Normalize(vmin=residual_ex[1].min(), vmax=residual_ex[1].max())
    plot_subgroup(example_idx=1, row_start=5+1, title='Fast rupture',
                  data_norm=norm2_data, truth_norm=norm2_truth, res_norm=norm2_res)

    # ---------- subgroup 3: rows 9–11 ----------
    norm3_data = TwoSlopeNorm(vmin=data_examples[2].min(), vcenter=0, vmax=data_examples[2].max())
    norm3_truth = TwoSlopeNorm(vmin=truth_examples[2].min(), vcenter=0, vmax=truth_examples[2].max())
    # norm3_res = TwoSlopeNorm(vmin=residual_ex[2].min(), vcenter=0, vmax=residual_ex[2].max())
    norm3_res = Normalize(vmin=residual_ex[2].min(), vmax=residual_ex[2].max())
    plot_subgroup(example_idx=2, row_start=9+2, title='Complex fault trace',
                  data_norm=norm3_data, truth_norm=norm3_truth, res_norm=norm3_res)'''
    # plt.show()
    plt.savefig('/Users/giuseppe/Documents/DATA/InSAR/insar-denoising/figures/supp/synthetic_two_events.pdf',
                bbox_inches='tight')


def plot_transects_synth_two_ev(x, y, pred):
    # segment = [[38, 93], [100, 60]]
    segment = [[38, 93-10], [60, 20]]


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
    x0 = bbox.x0 + 0.2 * bbox.width
    y0 = bbox.y0 + 0.2 * bbox.height
    w = 0.2 * bbox.width
    h = 0.03 * bbox.height

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


    ax_right_bottom.plot([i for i in range(9)], accy_d, label='Ground truth',
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
    plt.savefig('/Users/giuseppe/Documents/DATA/InSAR/insar-denoising/figures/supp/two_ev_transects.pdf',
                bbox_inches='tight')

if __name__ == '__main__':
    params = init_params()
    complex_noise, complex_topo = generate_noise_ts(params, seed=3359220284) # 1313731410 # 1983544845 # 1138737539 # 4045296993 # 3016178603
    _, slow1_y, topo1_slow = sample_rupture(params, onset_frame=1, slip=70., seed=3263276907)  # 616459241  # 3263276907
    _, slow2_y, _ = sample_rupture(params, onset_frame=6, slip=50., seed=3263276907)  # 616459241  # 3263276907

    complex_noise = np.astype(complex_noise, np.float32)
    slow1_y, slow2_y = np.astype(slow1_y, np.float32), np.astype(slow2_y, np.float32)
    topo1_slow = np.astype(topo1_slow, np.float32)

    # normalize topography
    topo1_slow = normalize_topo(topo1_slow)

    slow_y = slow1_y + slow2_y
    slow_x = complex_noise * 1. + slow_y


    pred = predict(denoiser_configuration, slow_x, slow_y, topo1_slow)

    slow_x = slow_x.squeeze()
    slow_y = slow_y.squeeze()
    topo_slow1 = topo1_slow.squeeze()
    pred = pred.squeeze()

    slow_x = slow_x[None,...]
    slow_y = slow_y[None, ...]
    pred = pred[None, ...]

    plot_results_two_ev_residuals(slow_x, slow_y, pred)

    slow_x = slow_x.squeeze()
    slow_y = slow_y.squeeze()
    pred = pred.squeeze()

    plot_transects_synth_two_ev(slow_x, slow_y, pred)