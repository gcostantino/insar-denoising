import matplotlib
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import TwoSlopeNorm

from config.local_denoiser_params import denoiser_configuration
from ismetpasa_denoising import load_creepmeter_data, predict, make_plot_3rows, plot_transects_new, \
    profile_along_segment, point_to_pixel_ref, compute_displacement_from_extrema


def plot_transects_new_float_labels(data, pred, lon, lat, point_lon, point_lat, np_creep_time, creep_data, creep_mask,
                                    datetimes,
                                    ismetpasa_strike, vmin=-2., vmax=1., save=True, float_cbar_left_label=True,
                                    float_cbar_right_label=True, ylims=None):
    # select_transects(img_pred, lon, lat, point_lon, point_lat)
    segment = [[112, 42], [109, 107]]
    segment = [[100, 20], [104, 100]]
    # segment = [[112, 50], [109, 90]]
    img_pred = pred[-1]
    img_data = data[-1] - data[0]

    ### median the pred image
    '''nan_mask = np.isnan(img_pred)
    img_pred[nan_mask] = 0.
    img_pred = uniform_filter(img_pred, 7)'''

    dist_pred, profile_pred = profile_along_segment(img_pred, segment[0], segment[1], n_samples=200, pixel_size=1.0)
    dist_data, profile_data = profile_along_segment(img_data, segment[0], segment[1], n_samples=200, pixel_size=1.0)
    '''dist_pred, profile_pred, profile_pred_std, *_ = profile_along_segment_std(img_pred, segment[0], segment[1], n_samples=200, pixel_size=1.0)
    dist_data, profile_data, profile_data_std, *_ = profile_along_segment_std(img_data, segment[0], segment[1], n_samples=200, pixel_size=1.0)'''
    px, py = point_to_pixel_ref(lon, lat, point_lon, point_lat, n_pixels=128)

    fig = plt.figure(figsize=(12, 6))

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
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax_left.imshow(
        img_pred,
        cmap='RdBu_r',
        norm=norm,
        origin='lower'
    )

    ax_left.scatter(px, py, s=100, c='green', marker='^', zorder=10., edgecolor='k')

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
    cbar.set_ticklabels([str((float if float_cbar_left_label else int)(norm.vmin)), str(norm.vcenter),
                         str((float if float_cbar_right_label else int)(norm.vmax))])

    ax_right_top.plot(dist_pred, profile_pred - profile_pred[13], color='C3', lw=2., label='Denoised')
    ax_right_top.plot([], [], color='k', lw=2., label='Data')
    ax_right_top.scatter(dist_data, profile_data - profile_data[13], color='k', s=5)

    # project creepmeter data in LOS
    # creep_data = surface_to_los(-creep_data, np.zeros_like(creep_data), np.zeros_like(creep_data), 33., -167.)
    head, inc = -167., 33.
    alpha = (head + 90.) * np.pi / 180.
    phi = inc * np.pi / 180.
    creep_data = - creep_data * np.sin(np.deg2rad(ismetpasa_strike)) * np.sin(alpha) * np.sin(
        phi) - creep_data * np.cos(np.deg2rad(ismetpasa_strike)) * np.sin(alpha) * np.sin(
        phi)  # # s_e = -1. * np.sin(alpha) * np.sin(phi), s_n = -1. * np.cos(alpha) * np.sin(phi)

    norm_creep_data = 0.1 * (creep_data[creep_mask] - creep_data[creep_mask][0])
    ax_right_bottom.plot(np_creep_time[creep_mask], norm_creep_data, lw=2., label='Creepmeter', zorder=2)
    ax_right_bottom.plot([], [], color='C3', lw=2., label='Denoised')

    ax_right_bottom.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=6))
    ax_right_bottom.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))

    acc_d, acc_d_mean, acc_d_std = [], [], []
    # plt.show()
    # predefined global min and max (None: unknown)
    min_idx, max_idx = [None] * 9, [None, None, None, 38 - 13, 37 - 13, None, None, None, None]
    for i, date in enumerate(datetimes):
        dd_i, pp_i = profile_along_segment(pred[i], segment[0], segment[1], n_samples=200, pixel_size=1.0)  # 128/2
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

        # acc_d.append(np.abs(pp_i[np.where((dd_i - 60)< 0.01)[0][0]] - pp_i[np.where((dd_i - 75)< 0.01)[0][0]]))
        acc_d.append(res['displacement'])
        acc_d_std.append(res['uncertainty'])
        print("res['displacement'] -->", res['displacement'])


    # acc_d = [0., 0.2, 0.9, 0.5, 0.25, 0.9, 1.5, 1.5, 2.5]
    # acc_d = [0., 0.0, 0.2, 0.5, .5, .8, .8, .9, 1.]
    ax_right_bottom.errorbar(datetimes, acc_d, yerr=acc_d_std,
                             fmt='o', color='C3', markersize=5,
                             capsize=3, capthick=1.,
                             zorder=1)

    # ax_right_bottom.set_xticklabels([])  # Remove x-axis labels

    # ax_prof.axvline(0, color='0.5', ls='--', lw=1)
    ax_right_top.set_xlabel('A-A\' distance along transect (pixels)', font='DejaVu Sans',
                            fontweight='bold')  # fontweight='bold', font='DejaVu Sans'
    ax_right_top.set_ylabel('Displacement [cm]', font='DejaVu Sans', fontweight='bold')
    ax_right_top.legend()
    ax_right_bottom.set_xlabel('Time', font='DejaVu Sans', fontweight='bold')
    ax_right_bottom.set_ylabel('Displacement [cm]', font='DejaVu Sans', fontweight='bold')
    ax_right_bottom.legend()

    ax_right_bottom.spines[['right', 'top']].set_visible(False)
    ax_right_top.spines[['right', 'top']].set_visible(False)

    if ylims:
        ax_right_bottom.set_ylim(ylims)

    # ax_prof.legend(loc='best', fontsize=8)
    # fig.tight_layout()
    if save:
        plt.savefig('./ismetpasa_transect.pdf',
                    bbox_inches='tight')
    else:
        return fig


def load_ts(every_nth=15):
    # rawts, _, topo, lon, lat, *_, velocity, datetimes, naf = prepare(track, lon0, lat0, plot=False)
    with np.load(f'/Users/giuseppe/Documents/DATA/InSAR/NAF/track_87_naf.npz', allow_pickle=True) as data:
        rawts, topo, lon, lat, velocity, datetimes = data['rawts'], data['topo'], data['lon'], data['lat'], data[
            'velocity'], data['datetimes']

    if velocity is None:
        velocity = rawts[-1] - rawts[0]

    rawts = 0.1 * rawts
    velocity = 0.1 * velocity

    # taking every 15 images
    rawts = rawts[::every_nth, ...]
    datetimes = datetimes[::every_nth]
    print('rawts.shape :', rawts.shape)

    rawtscpy = rawts.copy()
    nan_mask = np.isnan(rawtscpy)
    rnd_pixels = np.random.uniform(-10, 10, size=np.sum(nan_mask))
    # rawtscpy[nan_mask] = rnd_pixels
    rawts[nan_mask] = rnd_pixels

    normalized_topo = (topo - np.nanmin(topo)) / (np.nanmax(topo) - np.nanmin(topo))

    assert np.isnan(normalized_topo).sum() == 0
    assert np.min(normalized_topo) == 0
    assert np.max(normalized_topo) == 1

    return rawts, normalized_topo, lon, lat, velocity, datetimes, nan_mask


if __name__ == '__main__':
    #### supplementary data
    ### n.1 : every 22 acquisitions, increment = 0, t0, t1 = 1+increment, 10+increment (long time series)
    ### n.2 : every 2 acquisitions, t0=11, t1=20 (increment: 10)
    ### n.3 : every 3 acquisitions, increment = 26, t0, t1 = 1+increment, 10+increment (force ylim >=-1)

    ismetpasa_strike = 81.5  # np.rad2deg(np.arctan2(0.625, 0.09375))
    creep_obliquity = 33  # degrees

    np_creep_time, creep_data = load_creepmeter_data()
    ismetpasa_lat, ismetpasa_lon = 40.8698, 32.6258

    x0, x1 = 420, 548  # 411 -1, 539 - 1
    y0, y1 = 51, 179  # 34 - 4, 162 - 4

    every_nth_list = [22, 2, 3]
    increment_list = [0, 10, 26]
    norm_lim_list = [((-10, 1), (-2, 0.5)), ((-5, 5), (-0.5, 1)), ((-10, 1), (-0.5, 0.5))]
    float_left_list, float_right_list = [False, True, True], [True, False, True]
    ylim_list = [None, None, (-1, 1.)]
    cbar_data_ticklabels = [('-10', '-5', '0', '0.5'), None, ('-10', '-5', '0', '0.5')]
    cbar_pred_ticklabels = [('-2', '-1', '0', '0.5'), None, None]
    # cbar_gtp.set_ticklabels(['-3', '0', '5', '10'])

    creep_dates = [np.datetime64('2019-08-10'), np.datetime64('2017-01-01'), np.datetime64('2019-08-10')]

    for i, (every_nth, increment) in enumerate(zip(every_nth_list, increment_list)):
        rawts, topo, lon, lat, velocity, datetimes_arr, nan_mask = load_ts(every_nth=every_nth)
        print(velocity.shape, lat.shape, lon.shape, rawts.shape)
        t0, t1 = 1 + increment, 10 + increment
        # crop ts
        data = rawts[t0:t1, y0:y1, x0:x1]
        datetimes = datetimes_arr[t0:t1]

        nan_mask = nan_mask[t0:t1, y0:y1, x0:x1]
        lon_crop = lon[y0:y1, x0:x1]
        lat_crop = lat[y0:y1, x0:x1]

        topo = topo[y0:y1, x0:x1]
        # renormalize topography after crop
        topo = (topo - np.nanmin(topo)) / (np.nanmax(topo) - np.nanmin(topo))
        topo = topo[None, None, ..., None]

        data = data[None, ..., None]

        print('prima di predict:', data.shape, topo.shape)

        pred = predict(denoiser_configuration, data, data, topo)

        data, pred = data.squeeze(), pred.squeeze()

        data[nan_mask] = np.nan
        pred[nan_mask] = np.nan

        '''make_plot(velocity, lat, lon, data, pred, x0, x1, y0, y1, ismetpasa_lat, ismetpasa_lon, np_creep_time, creep_data,
                  nan_mask, lon_crop, lat_crop, datetimes)'''

        fig = make_plot_3rows(velocity, lat, lon, data, pred, x0, x1, y0, y1, ismetpasa_lat, ismetpasa_lon,
                              np_creep_time, creep_data,
                              nan_mask, lon_crop, lat_crop, datetimes, save=False, norm_lims=norm_lim_list[i],
                              linear_cbar=True, figsize=(16,16), cbar_data_ticklabels=cbar_data_ticklabels[i],
                              cbar_pred_ticklabels=cbar_pred_ticklabels[i])
        #plt.show()
        plt.savefig(f'./ismetpasa_supp_{i}.pdf',
                    bbox_inches='tight')
        # plot_transects(pred[-1])
        start_creep = np.datetime64('2015-05-27')
        # end_creep = np.datetime64('2018-04-11')
        end_creep = creep_dates[i]  # np.datetime64('2019-08-10')

        creep_mask = (np_creep_time >= start_creep) & (np_creep_time <= end_creep)
        selected_creep = creep_data[creep_mask]
        print('Total accumulated creep:', selected_creep[-1] - selected_creep[0])

        ref_idx = np.searchsorted(np_creep_time, datetimes[0])
        ref_value = creep_data[ref_idx]
        creep_data[ref_idx:] -= ref_value
        creep_data[:ref_idx] = 0.  # could be done better

        fig = plot_transects_new_float_labels(data, pred, lon_crop, lat_crop, ismetpasa_lon, ismetpasa_lat,
                                              np_creep_time,
                                              creep_data,
                                              creep_mask, datetimes, ismetpasa_strike, vmin=norm_lim_list[i][1][0],
                                              vmax=norm_lim_list[i][1][1], save=False,
                                              float_cbar_left_label=float_left_list[i],
                                              float_cbar_right_label=float_right_list[i], ylims=ylim_list[i])
        #plt.show()
        plt.savefig(f'./ismetpasa_profile_supp_{i}.pdf',
                    bbox_inches='tight')
