import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as scmap
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.dates import DateFormatter
from matplotlib.patches import Polygon
from scipy import stats
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.optimize import curve_fit

from config.local_denoiser_params import denoiser_configuration
from dataset.insarmemdataset import InSARMemDataset
from kito import GenericDataPipeline, Engine
from model.insar_denoiser import InSARDenoiser

SMALL_SIZE = 14
LEGEND_SIZE = 14
MEDIUM_SIZE = 16
LARGE_SIZE = 20
FONT_LIST = ['Helvetica', 'DejaVu Sans']

style_dict = {
    'font.size': SMALL_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_LIST,
    'axes.titlesize': LARGE_SIZE,
    'axes.labelsize': SMALL_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.titlesize': LARGE_SIZE
}

plt.rcParams.update(style_dict)

np.random.seed(42)


def zero_crossing_x(d, v):
    """
    Find x where v crosses 0 using linear interpolation.
    d, v are 1D arrays of same length.
    Returns np.nan if no sign change is found.
    """
    v = np.asarray(v)
    d = np.asarray(d)

    # indices where sign changes (…- +… or …+ -…) [web:243]
    idx = np.where(np.diff(np.signbit(v)))[0]
    if len(idx) == 0:
        return np.nan

    # here: take the first crossing; you can choose another if needed
    i = idx[0]
    x0, x1 = d[i], d[i + 1]
    y0, y1 = v[i], v[i + 1]

    # linear interpolation of zero crossing between the two samples [web:247][web:244]
    if y1 == y0:
        return x0  # degenerate, but avoids division by zero

    xz = x0 - y0 * (x1 - x0) / (y1 - y0)
    return xz


def select_transects(img, lon, lat, point_lon, point_lat):
    px, py = point_to_pixel_ref(lon, lat, point_lon, point_lat, n_pixels=128)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-1., vcenter=0, vmax=1.), origin='lower')  # img is 128x128
    ax.set_title('Click two points along the fault,\nthen press Enter')
    ax.scatter(px, py)
    plt.sca(ax)  # make sure this axes is current
    pts = plt.ginput(2, timeout=-1)  # blocking; returns [(x0,y0), (x1,y1)]
    plt.show()
    transect_points = []
    # [106  52] [119  94] (x0, y0), (x1, y1)
    # [88 59] [81 94] (x0, y0), (x1, y1)
    # [65 60] [50 92] (x0, y0), (x1, y1)

    # Convert to numpy arrays (in pixel coordinates)
    p0 = np.array(pts[0])  # (x0, y0)
    p1 = np.array(pts[1])  # (x1, y1)
    print("Fault endpoints (pixels):", p0, p1)


def rescale_segments(segments):
    segs = np.array(segments, dtype=float)  # shape (N, 2, 2) -> (N, (x,y) endpoints)
    p0 = segs[:, 0, :]  # (N,2)
    p1 = segs[:, 1, :]  # (N,2)

    vec = p1 - p0  # direction vectors
    L = np.linalg.norm(vec, axis=1)  # current lengths [web:217]
    Lmax = L.max()  # target length = longest transect

    u = vec / L[:, None]  # unit direction vectors [web:217]
    centers = 0.5 * (p0 + p1)  # midpoints

    halfL = Lmax / 2.0

    p0_new = centers - u * halfL
    p1_new = centers + u * halfL

    '''# optional: clip to image bounds to avoid going outside
    p0_new[:, 0] = np.clip(p0_new[:, 0], 0, W - 1)
    p0_new[:, 1] = np.clip(p0_new[:, 1], 0, H - 1)
    p1_new[:, 0] = np.clip(p1_new[:, 0], 0, W - 1)
    p1_new[:, 1] = np.clip(p1_new[:, 1], 0, H - 1)'''

    segments_equal = [tuple(map(tuple, pair)) for pair in np.stack([p0_new, p1_new], axis=1)]
    return segments_equal


def profile_along_segment(img, p0, p1, n_samples=200, pixel_size=1.0, order=1):
    """
    img: 2D array [y, x]
    p0, p1: (x, y) endpoints in pixel coords
    n_samples: samples along the segment
    pixel_size: physical length per pixel (optional)
    order: interpolation order for map_coordinates (1=bilinear)
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    vec = p1 - p0
    L = np.hypot(vec[0], vec[1])

    # parametric positions 0..1 along the line
    t = np.linspace(0, 1, n_samples)
    pts = p0[None, :] + t[:, None] * vec[None, :]

    x = pts[:, 0]
    y = pts[:, 1]

    # map_coordinates expects coords as [row=y, col=x] [web:160]
    vals = map_coordinates(img, [y, x], order=order, mode='nearest')

    dist = t * L * pixel_size
    return dist, vals


def profile_along_segment_std(img, p0, p1, n_samples=200, pixel_size=1.0,
                              swath_width=5, order=1):
    """
    img: 2D array [y, x]
    p0, p1: (x, y) endpoints in pixel coords
    n_samples: samples along the segment
    pixel_size: physical length per pixel (optional)
    swath_width: number of pixels perpendicular to profile for std calculation
    order: interpolation order for map_coordinates (1=bilinear)

    Returns:
        dist: distance along profile
        vals: mean values along profile
        stds: standard deviation perpendicular to profile
        mins: minimum values
        maxs: maximum values
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    vec = p1 - p0
    L = np.hypot(vec[0], vec[1])

    # Perpendicular vector (rotate 90 degrees)
    perp = np.array([-vec[1], vec[0]])
    perp = perp / np.linalg.norm(perp)  # normalize

    # Parametric positions 0..1 along the line
    t = np.linspace(0, 1, n_samples)

    # Sample multiple parallel lines
    offsets = np.linspace(-swath_width / 2, swath_width / 2, swath_width)

    all_vals = []
    for offset in offsets:
        # Points along this parallel line
        pts = p0[None, :] + t[:, None] * vec[None, :] + offset * perp[None, :]
        x = pts[:, 0]
        y = pts[:, 1]

        # Sample values
        vals = map_coordinates(img, [y, x], order=order, mode='nearest')
        all_vals.append(vals)

    all_vals = np.array(all_vals)  # shape: (swath_width, n_samples)

    # Compute statistics across swath (perpendicular to profile)
    vals_mean = np.mean(all_vals, axis=0)
    vals_std = np.std(all_vals, axis=0)
    vals_min = np.min(all_vals, axis=0)
    vals_max = np.max(all_vals, axis=0)

    dist = t * L * pixel_size

    return dist, vals_mean, vals_std, vals_min, vals_max


def load_ts():
    # rawts, _, topo, lon, lat, *_, velocity, datetimes, naf = prepare(track, lon0, lat0, plot=False)
    with np.load(f'/Users/giuseppe/Documents/DATA/InSAR/NAF/track_87_naf.npz', allow_pickle=True) as data:
        rawts, topo, lon, lat, velocity, datetimes = data['rawts'], data['topo'], data['lon'], data['lat'], data[
            'velocity'], data['datetimes']

    if velocity is None:
        velocity = rawts[-1] - rawts[0]

    rawts = 0.1 * rawts
    velocity = 0.1 * velocity

    # taking every 15 images
    rawts = rawts[::15, ...]
    datetimes = datetimes[::15]

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


def load_creepmeter_data():
    # creepmeter data
    creep_data = np.loadtxt('/Users/giuseppe/Documents/DATA/creepmeter/NAF/IsmetpasaWall North 9Oct2019.txt',
                            delimiter=',', usecols=1)
    creep_time = np.loadtxt('/Users/giuseppe/Documents/DATA/creepmeter/NAF/IsmetpasaWall North 9Oct2019.txt',
                            delimiter=',', usecols=0, dtype=str)
    creep_time = np.array([datetime.datetime.strptime(str_time, '%m/%d/%Y %H:%M:%S') for str_time in creep_time])

    np_creep_time = np.array(creep_time, dtype="datetime64[ms]")
    return np_creep_time, creep_data


def make_plot(vel, lat, lon, data, pred, x0, x1, y0, y1, ismetpasa_lat, ismetpasa_lon, np_creep_time, creep_data,
              nan_mask, lon_crop, lat_crop, datetimes):
    # cmap = 'RdBu_r'
    cmap_data = scmap.roma_r
    cmap_pred = scmap.vik
    # build quadrilateral from the actual pcolormesh cell corners
    pts = np.array([
        [lon[y0, x0], lat[y0, x0]],  # top-left
        [lon[y0, x1], lat[y0, x1]],  # top-right
        [lon[y1, x1], lat[y1, x1]],  # bottom-right
        [lon[y1, x0], lat[y1, x0]],  # bottom-left
    ])

    poly = Polygon(pts, closed=True, fill=False,
                   edgecolor='k', linewidth=1)

    # interval to highlight in creepmeter data
    start_creep = np.datetime64('2015-05-27')
    end_creep = np.datetime64('2018-04-11')

    fig = plt.figure(figsize=(16, 10))

    # ---- OUTER GRID: 2 vertical parts ----
    outer = fig.add_gridspec(nrows=2, ncols=1,
                             height_ratios=[1, 2],
                             hspace=0.2)

    # ====================================================
    # TOP PART: 1 row, 3 columns logically (2+1)
    #    - left axes spans first 2 columns
    #    - right axes uses 3rd column
    # ====================================================
    gs_top = outer[0].subgridspec(1, 3, wspace=0.3)

    ax_top_left = fig.add_subplot(gs_top[0, 0:2])  # spans 2 columns
    ax_top_right = fig.add_subplot(gs_top[0, 2])  # single column

    vel_norm = TwoSlopeNorm(vmin=np.nanmin(vel), vcenter=0., vmax=np.nanmax(vel))
    im_top_left = ax_top_left.pcolormesh(lon, lat, vel, norm=vel_norm, cmap=scmap.roma_r)
    ax_top_right.plot(np_creep_time, (creep_data - creep_data[0]) * 0.1)

    central_lat = np.mean(lat)
    km_per_degree = 111 * np.cos(np.radians(central_lat))

    scale_length_degrees = 20 / km_per_degree

    # Position for scale bar (in data coordinates)
    x_start = lon.min() + 0.4 * (lon.max() - lon.min())
    y_pos = lat.min() + 0.05 * (lat.max() - lat.min())

    # Draw the scale bar
    ax_top_left.plot([x_start, x_start + scale_length_degrees],
                     [y_pos, y_pos],
                     color='black',
                     linewidth=3,
                     solid_capstyle='butt')

    # Add text label
    ax_top_left.text(x_start + scale_length_degrees / 2,
                     y_pos + 0.1 * (lat.max() - lat.min()),
                     '20 km',
                     ha='center',
                     va='top',
                     fontsize=14,
                     weight='bold')

    # grey shaded area between the two dates
    ax_top_right.axvspan(start_creep, end_creep, color='0.8', alpha=0.4, zorder=-1)  # zorder optional

    ax_top_left.add_patch(poly)

    ax_top_left.scatter(ismetpasa_lon, ismetpasa_lat, marker='^', color='k', s=10)

    for date in datetimes:
        ax_top_right.axvline(date, color='grey', linewidth=2., linestyle='--', alpha=.5)  # acquisitions
        # ax_top_right.axvline(date, color='k', linewidth=2., ymax=.05)

    ax_top_left.spines[['right', 'top']].set_visible(False)
    ax_top_right.spines[['right', 'top']].set_visible(False)

    ax_top_left.set_xlabel("Longitude")
    ax_top_left.set_ylabel("Latitude")
    ax_top_right.set_ylabel('Accumulated creep [cm]')
    ax_top_right.set_xlabel('Time [yr]')

    fig.canvas.draw()  # make sure positions are known

    bbox = ax_top_left.get_position()  # Bbox in figure coords [web:278]

    x0 = bbox.x0 + 0.7 * bbox.width
    y0 = bbox.y0 + 0.25 * bbox.height
    w = 0.15 * bbox.width
    h = 0.05 * bbox.height

    cax = fig.add_axes([x0, y0, w, h])  # explicit coordinates [web:271][web:272]

    cbar = fig.colorbar(im_top_left, cax=cax, orientation='horizontal')
    cbar.set_label('LOS velocity [cm/yr]')
    cax.xaxis.set_ticks_position('bottom')

    # LOS arrow parameters
    angle_deg = 33.0
    theta = np.deg2rad(angle_deg)  # radians
    L = 0.1  # arrow length in axes units (tune)

    # Start point in AXES coordinates (0–1, 0–1)
    x0, y0 = 0.1, 0.6  # tune to place arrow where you want

    dx = L * np.cos(theta)
    dy = L * np.sin(theta)

    ax_top_left.annotate(
        '',  # no text, just arrow
        xy=(x0 + dx, y0 + dy),  # arrow head
        xytext=(x0, y0),  # tail
        xycoords=ax_top_left.transAxes,  # interpret x0,y0 in axes coords
        arrowprops=dict(
            arrowstyle='-|>',  # filled arrow head
            lw=2,
            color='k'
        )
    )
    p0_disp = ax_top_left.transAxes.transform((x0, y0))
    p1_disp = ax_top_left.transAxes.transform((x0 + dx, y0 + dy))
    d_disp = p1_disp - p0_disp
    angle_screen = np.degrees(np.arctan2(d_disp[1], d_disp[0]))  # true on-screen angle

    # Midpoint of the arrow in axes coordinates
    xm = x0 + 0.45 * dx
    ym = y0 + 0.5 * dy + 0.05

    ax_top_left.text(
        xm, ym, 'LOS',
        transform=ax_top_left.transAxes,
        ha='center', va='center',
        rotation=angle_screen,
        rotation_mode='anchor',  # rotate around the text center [web:306][web:316]
        fontsize=10,
        color='k'
    )

    # ====================================================
    # BOTTOM PART: 2 rows × 9 columns
    # ====================================================
    gs_bottom = outer[1].subgridspec(2, 9, wspace=0.2, hspace=-0.5)

    data_to_plot = [data, pred]
    cmap_list = [cmap_data, cmap_pred]
    norm_lims = [(-5., 5.), (-2., 1.)]

    axes_bottom = []
    im_list = []
    for i in range(2):
        row_axes = []
        norm = TwoSlopeNorm(vmin=norm_lims[i][0], vcenter=0, vmax=norm_lims[i][1])
        # norm = TwoSlopeNorm(vmin=-1., vcenter=0, vmax=1.)
        for j in range(9):
            ax = fig.add_subplot(gs_bottom[i, j])
            im_bottom = ax.imshow(data_to_plot[i][j], cmap=cmap_list[i], norm=norm, origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            row_axes.append(ax)

            if i == 0:
                ax.set_title(datetimes[j].strftime('%d %b %Y'), fontsize=13)  # titles for upper row
        axes_bottom.append(row_axes)
        im_list.append(im_bottom)

    # Shared colorbar for ALL bottom subplots at far right
    fig.canvas.draw()

    # ----------------------------------------------------
    # TWO COLORBARS FOR BOTTOM BLOCK (one per row)
    # ----------------------------------------------------
    fig.canvas.draw()

    # Rightmost axes of each row (for x/y extents)
    ax_r0 = axes_bottom[0][-1]
    ax_r1 = axes_bottom[1][-1]

    b0 = ax_r0.get_position()
    b1 = ax_r1.get_position()

    x_cbar_bot = b0.x1 + 0.01  # same x for both
    w_cbar_bot = 0.015 / 2

    # Row 0 colorbar: aligned with first bottom row only
    y0_row0 = b0.y0
    h_row0 = b0.height

    cax_row0 = fig.add_axes([x_cbar_bot, y0_row0, w_cbar_bot, h_row0])
    cbar_row0 = fig.colorbar(im_list[0], cax=cax_row0)
    cbar_row0.set_label("LOS disp. [cm]")

    # Row 1 colorbar: aligned with second bottom row only
    y0_row1 = b1.y0
    h_row1 = b1.height

    cax_row1 = fig.add_axes([x_cbar_bot, y0_row1, w_cbar_bot, h_row1])
    cbar_row1 = fig.colorbar(im_list[1], cax=cax_row1)
    cbar_row1.set_label("LOS disp. [cm]")

    plt.savefig('./ismetpasa_denoising.pdf',
                bbox_inches='tight')


def make_plot_3rows(vel, lat, lon, data, pred, x0, x1, y0, y1, ismetpasa_lat, ismetpasa_lon, np_creep_time, creep_data,
                    nan_mask, lon_crop, lat_crop, datetimes):
    # cmap = 'RdBu_r'
    cmap_data = scmap.roma_r
    cmap_pred = scmap.vik
    # build quadrilateral from the actual pcolormesh cell corners
    pts = np.array([
        [lon[y0, x0], lat[y0, x0]],  # top-left
        [lon[y0, x1], lat[y0, x1]],  # top-right
        [lon[y1, x1], lat[y1, x1]],  # bottom-right
        [lon[y1, x0], lat[y1, x0]],  # bottom-left
    ])

    poly = Polygon(pts, closed=True, fill=False,
                   edgecolor='k', linewidth=1)

    # interval to highlight in creepmeter data
    start_creep = np.datetime64('2015-05-27')
    end_creep = np.datetime64('2018-04-11')

    fig = plt.figure(figsize=(16, 16))

    # ---- OUTER GRID: 2 vertical parts ----
    outer = fig.add_gridspec(nrows=2, ncols=1,
                             height_ratios=[3, 2],
                             hspace=0.2)

    # ====================================================
    # TOP PART: 1 row, 3 columns logically (2+1)
    #    - left axes spans first 2 columns
    #    - right axes uses 3rd column
    # ====================================================
    gs_top = outer[0].subgridspec(3, 1, wspace=0.3, hspace=0.5)

    '''ax_top_left = fig.add_subplot(gs_top[0, 0:2])  # spans 2 columns
    ax_top_right = fig.add_subplot(gs_top[0, 2])  # single column'''
    ax_top_up = fig.add_subplot(gs_top[0:2, 0])  # spans 2 columns
    ax_top_down = fig.add_subplot(gs_top[2, 0])  # single column

    vel_norm = TwoSlopeNorm(vmin=np.nanmin(vel), vcenter=0., vmax=np.nanmax(vel))
    im_top_left = ax_top_up.pcolormesh(lon, lat, vel, norm=vel_norm, cmap=scmap.roma_r)
    ax_top_down.plot(np_creep_time, (creep_data - creep_data[0]) * 0.1)

    central_lat = np.mean(lat)
    km_per_degree = 111 * np.cos(np.radians(central_lat))

    scale_length_degrees = 20 / km_per_degree

    # Position for scale bar (in data coordinates)
    x_start = lon.min() + 0.4 * (lon.max() - lon.min())
    y_pos = lat.min() + 0.05 * (lat.max() - lat.min())

    # Draw the scale bar
    ax_top_up.plot([x_start, x_start + scale_length_degrees],
                   [y_pos, y_pos],
                   color='black',
                   linewidth=3,
                   solid_capstyle='butt')

    # Add text label
    ax_top_up.text(x_start + scale_length_degrees / 2,
                   y_pos + 0.1 * (lat.max() - lat.min()),
                   '20 km',
                   ha='center',
                   va='top',
                   fontsize=14,
                   weight='bold')

    # grey shaded area between the two dates
    ax_top_down.axvspan(start_creep, end_creep, color='0.8', alpha=0.4, zorder=-1)  # zorder optional

    ax_top_up.add_patch(poly)

    ax_top_up.scatter(ismetpasa_lon, ismetpasa_lat, marker='^', color='k', s=10)

    for date in datetimes:
        ax_top_down.axvline(date, color='grey', linewidth=2., linestyle='--', alpha=.5)  # acquisitions
        # ax_top_right.axvline(date, color='k', linewidth=2., ymax=.05)

    ax_top_up.spines[['right', 'top']].set_visible(False)
    ax_top_down.spines[['right', 'top']].set_visible(False)

    ax_top_up.set_xlabel("Longitude", fontweight='bold', font='DejaVu Sans')
    ax_top_up.set_ylabel("Latitude", fontweight='bold', font='DejaVu Sans')
    ax_top_down.set_ylabel('Accumulated creep [cm]', fontweight='bold', font='DejaVu Sans')
    ax_top_down.set_xlabel('Time [yr]', fontweight='bold', font='DejaVu Sans')

    fig.canvas.draw()  # make sure positions are known

    bbox = ax_top_up.get_position()  # Bbox in figure coords [web:278]

    x0 = bbox.x0 + 0.7 * bbox.width
    y0 = bbox.y0 + 0.25 * bbox.height
    w = 0.15 * bbox.width
    h = 0.05 * bbox.height

    cax = fig.add_axes([x0, y0, w, h])  # explicit coordinates [web:271][web:272]

    cbar = fig.colorbar(im_top_left, cax=cax, orientation='horizontal')
    cbar.set_label('LOS velocity [cm/yr]')
    cax.xaxis.set_ticks_position('bottom')

    # LOS arrow parameters
    angle_deg = 33.0
    theta = np.deg2rad(angle_deg)  # radians
    L = 0.1  # arrow length in axes units (tune)

    # Start point in AXES coordinates (0–1, 0–1)
    x0, y0 = 0.1, 0.6  # tune to place arrow where you want

    dx = L * np.cos(theta)
    dy = L * np.sin(theta)

    ax_top_up.annotate(
        '',  # no text, just arrow
        xy=(x0 + dx, y0 + dy),  # arrow head
        xytext=(x0, y0),  # tail
        xycoords=ax_top_up.transAxes,  # interpret x0,y0 in axes coords
        arrowprops=dict(
            arrowstyle='-|>',  # filled arrow head
            lw=2,
            color='k'
        )
    )
    p0_disp = ax_top_up.transAxes.transform((x0, y0))
    p1_disp = ax_top_up.transAxes.transform((x0 + dx, y0 + dy))
    d_disp = p1_disp - p0_disp
    angle_screen = np.degrees(np.arctan2(d_disp[1], d_disp[0]))  # true on-screen angle

    # Midpoint of the arrow in axes coordinates
    xm = x0 + 0.45 * dx
    ym = y0 + 0.5 * dy + 0.05

    ax_top_up.text(
        xm, ym, 'LOS',
        transform=ax_top_up.transAxes,
        ha='center', va='center',
        rotation=angle_screen,
        rotation_mode='anchor',  # rotate around the text center [web:306][web:316]
        fontsize=10,
        color='k'
    )

    # ====================================================
    # BOTTOM PART: 2 rows × 9 columns
    # ====================================================
    gs_bottom = outer[1].subgridspec(2, 9, wspace=0.2,
                                     hspace=-0.5)  # outer[1].subgridspec(2, 9, wspace=0.2, hspace=-0.5)

    data_to_plot = [data, pred]
    cmap_list = [cmap_data, cmap_pred]
    norm_lims = [(-5., 5.), (-2., 1.)]

    axes_bottom = []
    im_list = []
    for i in range(2):
        row_axes = []
        norm = TwoSlopeNorm(vmin=norm_lims[i][0], vcenter=0, vmax=norm_lims[i][1])
        # norm = TwoSlopeNorm(vmin=-1., vcenter=0, vmax=1.)
        for j in range(9):
            ax = fig.add_subplot(gs_bottom[i, j])
            im_bottom = ax.imshow(data_to_plot[i][j], cmap=cmap_list[i], norm=norm, origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            row_axes.append(ax)

            if i == 0:
                ax.set_title(datetimes[j].strftime('%d %b %Y'), fontsize=13)  # titles for upper row
        axes_bottom.append(row_axes)
        im_list.append(im_bottom)

    # Shared colorbar for ALL bottom subplots at far right
    fig.canvas.draw()

    # ----------------------------------------------------
    # TWO COLORBARS FOR BOTTOM BLOCK (one per row)
    # ----------------------------------------------------
    fig.canvas.draw()

    # Rightmost axes of each row (for x/y extents)
    ax_r0 = axes_bottom[0][-1]
    ax_r1 = axes_bottom[1][-1]

    b0 = ax_r0.get_position()
    b1 = ax_r1.get_position()

    x_cbar_bot = b0.x1 + 0.01  # same x for both
    w_cbar_bot = 0.015 / 2

    # Row 0 colorbar: aligned with first bottom row only
    y0_row0 = b0.y0
    h_row0 = b0.height

    cax_row0 = fig.add_axes([x_cbar_bot, y0_row0, w_cbar_bot, h_row0])
    cbar_row0 = fig.colorbar(im_list[0], cax=cax_row0)
    cbar_row0.set_label("LOS disp. [cm]")

    # Row 1 colorbar: aligned with second bottom row only
    y0_row1 = b1.y0
    h_row1 = b1.height

    cax_row1 = fig.add_axes([x_cbar_bot, y0_row1, w_cbar_bot, h_row1])
    cbar_row1 = fig.colorbar(im_list[1], cax=cax_row1)
    cbar_row1.set_label("LOS disp. [cm]")

    plt.savefig('./ismetpasa_denoising.pdf',
                bbox_inches='tight')


def predict(denoiser_configuration, x, y, topo):
    denoiser_configuration.data.total_samples = len(x)
    denoiser_configuration.training.batch_size = 1
    denoiser_configuration.training.train_data_ratio = 0.
    denoiser_configuration.training.val_data_ratio = 0.

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


def plot_transects(img):
    # select_transects(img)

    segments = [
        # ((106, 52), (119, 94)),  # (x0, y0) -> (x1, y1)
        ((107, 55), (110, 95)),
        ((88, 59), (81, 94)),
        ((65, 60), (50, 92)),
        ((97, 54), (90, 99)),
        ((54, 58), (30, 83)),
        ((36, 49), (16, 74)),
    ]
    segments = rescale_segments(segments)
    # [97 54] [90 99]
    profiles = []
    dists = []

    for (p0, p1) in segments:
        d, v = profile_along_segment(img, p0, p1, n_samples=200, pixel_size=1.0)
        dists.append(d)
        profiles.append(v)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'wspace': 0.35})

    # -------------------------------------------------
    # LEFT: image with transects
    # -------------------------------------------------
    norm = TwoSlopeNorm(vmin=-2., vcenter=0, vmax=1.)
    ax_img = axes[0]
    im = ax_img.imshow(
        img,
        cmap='RdBu_r',
        norm=norm,
        origin='lower'
    )

    for (p0, p1), c in zip(segments, colors):
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        ax_img.plot(xs, ys, color=c, lw=2)

    ax_img.set_aspect('equal')
    # ax_img.set_title('Cumulative deformation')
    ax_img.set_xlabel('Longitude (pixels)')
    ax_img.set_ylabel('Latitude (pixels)')

    fig.canvas.draw()  # make sure positions are known

    bbox = ax_img.get_position()  # Bbox in figure coords [web:278]

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

    # -------------------------------------------------
    # RIGHT: displacement profiles
    # -------------------------------------------------
    ax_prof = axes[1]
    '''for k, (d, v) in enumerate(zip(dists, profiles)):
        ax_prof.plot(d, v, color=colors[k], lw=2, label=f'Transect {k + 1}')'''

    # ---- right: profiles shifted so zero-crossing is at x=0 ----
    for k, (d, v, c) in enumerate(zip(dists, profiles, colors)):
        xz = zero_crossing_x(d, v)  # position where v crosses 0 [web:233][web:243]
        if np.isnan(xz):
            x_shift = d  # no zero-crossing found
        else:
            x_shift = d - xz  # shift so crossing is at 0

        ax_prof.plot(x_shift, v, color=c, lw=2, label=f'Transect {k + 1}')

    # ax_prof.axvline(0, color='0.5', ls='--', lw=1)
    ax_prof.set_xlabel('Distance along transect (pixels)')
    ax_prof.set_ylabel('Displacement [cm]')
    # ax_prof.legend(loc='best', fontsize=8)
    # fig.tight_layout()
    plt.savefig('./ismetpasa_transects.pdf',
                bbox_inches='tight')


def point_to_pixel_ref(lon, lat, point_lon, point_lat, n_pixels=128):
    # Create pixel coordinate arrays
    pixel_y, pixel_x = np.meshgrid(np.arange(n_pixels), np.arange(n_pixels), indexing='ij')

    pixel_y_flat = pixel_y.flatten()
    pixel_x_flat = pixel_x.flatten()

    lat_flat = lat.flatten()
    lon_flat = lon.flatten()

    print(pixel_x_flat.shape, lon.shape)

    # Use nearest neighbor or linear interpolation
    px = griddata((lon_flat, lat_flat), pixel_x_flat,
                  (point_lon, point_lat), method='nearest')
    py = griddata((lon_flat, lat_flat), pixel_y_flat,
                  (point_lon, point_lat), method='nearest')
    return px, py


def point_coords_on_profile(dist, profile, px, py):
    # Step 1: Project point onto the segment to find its distance along profile
    p0 = np.array(profile[0], dtype=float)
    p1 = np.array(profile[1], dtype=float)
    point = np.array([px, py], dtype=float)

    # Vector along segment
    v = p1 - p0
    # Vector from start to point
    w = point - p0

    # Find parametric position (0 to 1) along segment
    t = np.dot(w, v) / np.dot(v, v)
    t = np.clip(t, 0, 1)  # Keep on segment

    # Distance of point along the profile
    point_distance = t * np.linalg.norm(v)

    # Step 2: Interpolate profile value at this distance
    profile_value_at_point = np.interp(point_distance, dist, profile)
    return profile_value_at_point


def get_profile_value_with_dispersion(point, segment, dist_profile, values_profile, window=5):
    """
    Get profile value at point with dispersion from neighboring points.

    Parameters:
    -----------
    point: (px, py) point coordinates
    segment: [(x0,y0), (x1,y1)] segment endpoints
    dist_profile: distances along profile
    values_profile: values along profile
    window: number of points on each side (default 5)

    Returns:
    --------
    value_at_point: interpolated value at the point
    mean_local: mean of nearby points
    std_local: standard deviation of nearby points
    point_dist: distance along profile
    """
    # Find where point projects onto profile
    p0, p1 = np.array(segment[0]), np.array(segment[1])
    pt = np.array(point)

    v = p1 - p0
    t = np.clip(np.dot(pt - p0, v) / np.dot(v, v), 0, 1)
    point_dist = t * np.linalg.norm(v)

    # Find nearest index in profile
    nearest_idx = np.argmin(np.abs(dist_profile - point_dist))

    # Extract window around this point
    idx_start = max(0, nearest_idx - window)
    idx_end = min(len(values_profile), nearest_idx + window + 1)

    local_values = values_profile[idx_start:idx_end]

    # Compute statistics
    value_at_point = np.interp(point_dist, dist_profile, values_profile)
    mean_local = np.mean(local_values)
    std_local = np.std(local_values)
    min_local = np.min(local_values)
    max_local = np.max(local_values)

    return value_at_point, mean_local, std_local, min_local, max_local, point_dist


def arctangent_model(x, d0, x0, w, offset):
    """
    Arctangent model for fault displacement profile.

    d0: total displacement (what you want!)
    x0: position of fault
    w: width of transition zone
    offset: background offset
    """
    return (d0 / np.pi) * np.arctan((x - x0) / w) + offset


def compute_static_displacement_model(dist, profile):
    """Fit arctangent and extract displacement."""
    # Initial guess
    d0_guess = np.percentile(profile, 95) - np.percentile(profile, 5)
    x0_guess = dist[len(dist) // 2]
    w_guess = (dist[-1] - dist[0]) / 10
    offset_guess = np.median(profile)

    p0 = [d0_guess, x0_guess, w_guess, offset_guess]
    print('p0', p0)

    try:
        popt, pcov = curve_fit(arctangent_model, dist, profile, p0=p0, nan_policy='omit')
        d0_fit, x0_fit, w_fit, offset_fit = popt
        print('popt', popt)
        # Uncertainty from covariance
        d0_std = np.sqrt(np.diag(pcov))[0]

        return d0_fit, d0_std, popt
    except:
        return np.nan, np.nan, None


def compute_displacement_from_extrema_old(dist, profile, delta_x=5, smooth_sigma=2., min_idx=None, max_idx=None,
                                          bilateral=False, median=False):
    """
    Find minimum and maximum in profile, compute values in windows around them.

    Parameters:
    -----------
    dist: distance array
    profile: profile values
    delta_x: window size (±delta_x) around min/max for averaging
    smooth_sigma: gaussian smoothing parameter (0 = no smoothing)

    Returns:
    --------
    displacement: max_value - min_value
    min_value, max_value: computed values at extrema
    min_pos, max_pos: positions of min and max
    min_std, max_std: standard deviations in windows
    """

    # Step 1: Smooth data if needed
    if smooth_sigma > 0:
        profile_smooth = gaussian_filter1d(profile, sigma=smooth_sigma)
    else:
        profile_smooth = profile.copy()

    # Step 2: Find global minimum and maximum
    if min_idx is None:
        min_idx = np.argmin(profile_smooth)
    if max_idx is None:
        max_idx = np.argmax(profile_smooth)

    min_pos = dist[min_idx]
    max_pos = dist[max_idx]
    if bilateral:
        # Step 3: Define windows around min and max
        # Find indices within delta_x distance
        min_window_mask = np.abs(dist - min_pos) <= delta_x
        max_window_mask = np.abs(dist - max_pos) <= delta_x

        # Extract values in windows (use original, not smoothed)
        min_window_vals = profile[min_window_mask]
        max_window_vals = profile[max_window_mask]
    else:
        # Step 3: Define one-sided windows around min and max

        # For minimum: from min to min + delta_x (right side only)
        min_window_mask = (dist >= min_pos) & (dist <= min_pos + delta_x)

        # For maximum: from max - delta_x to max (left side only)
        max_window_mask = (dist >= max_pos - delta_x) & (dist <= max_pos)

        # Extract values in windows (use original, not smoothed)
        min_window_vals = profile[min_window_mask]
        max_window_vals = profile[max_window_mask]

    # Step 4: Compute statistics in windows
    if median:
        min_value = np.median(min_window_vals)
        max_value = np.median(max_window_vals)

        min_mad = np.median(np.abs(min_window_vals - min_value))
        max_mad = np.median(np.abs(max_window_vals - max_value))

        # Displacement
        displacement = max_value - min_value
        displacement_std = np.sqrt(min_mad ** 2 + max_mad ** 2)
    else:
        min_value = np.mean(min_window_vals)
        max_value = np.mean(max_window_vals)
        min_std = np.std(min_window_vals)
        max_std = np.std(max_window_vals)

        # Displacement
        displacement = max_value - min_value
        displacement_std = np.sqrt(min_std ** 2 + max_std ** 2)

    return {
        'displacement': displacement,
        'uncertainty': displacement_std,
        'min_value': min_value,
        'max_value': max_value,
        'min_pos': min_pos,
        'max_pos': max_pos,
        'min_unc': min_mad if median else min_std,
        'max_unc': max_mad if median else max_std,
        'min_window_mask': min_window_mask,
        'max_window_mask': max_window_mask
    }


def compute_displacement_from_extrema(dist, profile, delta_x=5, smooth_sigma=2., min_idx=None, max_idx=None,
                                      bilateral=False, median=False, method='mean'):
    """
    Find minimum and maximum in profile, compute values in windows around them.

    Parameters:
    -----------
    dist: distance array
    profile: profile values
    delta_x: window size (±delta_x) around min/max for averaging
    smooth_sigma: gaussian smoothing parameter (0 = no smoothing)
    min_idx, max_idx: optional fixed indices for min/max
    bilateral: if True, use symmetric windows; if False, use one-sided
    median: if True, use median/MAD; if False, use mean/std
    method: 'mean' (average in window), 'median', or 'linear' (fit lines before/after)

    Returns:
    --------
    displacement: max_value - min_value (or offset between fitted lines)
    min_value, max_value: computed values at extrema
    min_pos, max_pos: positions of min and max
    min_unc, max_unc: uncertainties
    """

    # remove nans
    valid_data_mask = ~np.isnan(profile)
    profile = profile[valid_data_mask]

    # Step 1: Smooth data if needed
    if smooth_sigma > 0:
        profile_smooth = gaussian_filter1d(profile, sigma=smooth_sigma)
    else:
        profile_smooth = profile.copy()

    # Step 2: Find global minimum and maximum
    if min_idx is None:
        min_idx = np.argmin(profile_smooth)
    if max_idx is None:
        max_idx = np.argmax(profile_smooth)

    min_pos = dist[min_idx]
    max_pos = dist[max_idx]

    # Step 3: Parse delta_x input
    if isinstance(delta_x, (tuple, list)):
        delta_main, delta_transition = delta_x
    else:
        delta_main = delta_x
        delta_transition = 0  # No transition zone by default

    # Define windows based on bilateral flag
    if bilateral:
        # Symmetric windows (ignore transition parameter)
        min_window_mask = np.abs(dist - min_pos) <= delta_main
        max_window_mask = np.abs(dist - max_pos) <= delta_main
    else:
        # Asymmetric windows with transition zone
        # Min window: main goes right, transition goes left
        min_window_mask = (dist >= min_pos - delta_transition) & (dist <= min_pos + delta_main)

        # Max window: main goes left, transition goes right
        max_window_mask = (dist >= max_pos - delta_main) & (dist <= max_pos + delta_transition)

    # Extract values in windows (use original, not smoothed)
    min_window_vals = profile[min_window_mask]
    max_window_vals = profile[max_window_mask]
    min_window_dist = dist[min_window_mask]
    max_window_dist = dist[max_window_mask]

    # Step 3: Compute statistics based on method
    if method == 'linear':
        # Fit linear regressions in both windows
        # For minimum window (typically "before" the feature)
        if len(min_window_vals) >= 2:
            slope_min, intercept_min, r_min, p_min, stderr_min = stats.linregress(min_window_dist, min_window_vals)
            min_fit = slope_min * min_window_dist + intercept_min
            min_residuals = min_window_vals - min_fit
            min_value = slope_min * min_pos + intercept_min  # Evaluate at min_pos
            min_unc = np.std(min_residuals) if len(min_residuals) > 1 else 0.0
        else:
            min_value = np.mean(min_window_vals)
            min_unc = 0.0

        # For maximum window (typically "after" the feature)
        if len(max_window_vals) >= 2:
            slope_max, intercept_max, r_max, p_max, stderr_max = stats.linregress(max_window_dist, max_window_vals)
            max_fit = slope_max * max_window_dist + intercept_max
            max_residuals = max_window_vals - max_fit
            max_value = slope_max * max_pos + intercept_max  # Evaluate at max_pos
            max_unc = np.std(max_residuals) if len(max_residuals) > 1 else 0.0
        else:
            max_value = np.mean(max_window_vals)
            max_unc = 0.0

        plt.plot(dist, profile, 'k')
        # plt.plot(dist, profile_smooth)
        plt.plot(min_window_dist, min_window_vals, color='red')
        plt.plot(min_window_dist, min_fit, '--', color='red')
        plt.plot(max_window_dist, max_window_vals, color='green')
        plt.plot(max_window_dist, max_fit, '--', color='green')
        plt.show()

        # Evaluate at midpoint
        # eval_pos = (min_pos + max_pos) / 2
        # min_value_at_eval = slope_min * eval_pos + intercept_min
        # max_value_at_eval = slope_max * eval_pos + intercept_max
        # displacement = max_value_at_eval - min_value_at_eval

        displacement = max_value - min_value
        displacement_std = np.sqrt(min_unc ** 2 + max_unc ** 2)

    elif median or method == 'median':
        min_value = np.median(min_window_vals)
        max_value = np.median(max_window_vals)
        min_unc = np.median(np.abs(min_window_vals - min_value))
        max_unc = np.median(np.abs(max_window_vals - max_value))
        displacement = max_value - min_value
        displacement_std = np.sqrt(min_unc ** 2 + max_unc ** 2)

    else:  # method == 'mean' (default)
        min_value = np.mean(min_window_vals)
        max_value = np.mean(max_window_vals)
        min_unc = np.std(min_window_vals)
        max_unc = np.std(max_window_vals)
        displacement = max_value - min_value
        displacement_std = np.sqrt(min_unc ** 2 + max_unc ** 2)

    return {
        'displacement': displacement,
        'uncertainty': displacement_std,
        'min_value': min_value,
        'max_value': max_value,
        'min_pos': min_pos,
        'max_pos': max_pos,
        'min_unc': min_unc,
        'max_unc': max_unc,
        'min_window_mask': min_window_mask,
        'max_window_mask': max_window_mask,
        'method': method,
    }


def plot_displacement_analysis_old(dist, profile, result, frame_n, profile_smooth=None):
    """
    Plot profile with extrema and window regions.

    Parameters:
    -----------
    dist: distance array
    profile: profile values
    result: dictionary output from compute_displacement_from_extrema
    profile_smooth: optional smoothed profile
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot original profile
    ax.plot(dist, profile, 'o-', color='gray', alpha=0.6,
            markersize=3, linewidth=1, label='Profile')

    # Plot smoothed profile if available
    if profile_smooth is not None:
        ax.plot(dist, profile_smooth, 'k--', linewidth=2,
                alpha=0.7, label='Smoothed')

    # Shaded bands for min and max windows
    y_min, y_max = ax.get_ylim()

    ax.fill_between(dist, y_min, y_max,
                    where=result['min_window_mask'],
                    alpha=0.3, color='red',
                    label=f'Min window (±{result["min_unc"]:.3f})')

    ax.fill_between(dist, y_min, y_max,
                    where=result['max_window_mask'],
                    alpha=0.3, color='green',
                    label=f'Max window (±{result["max_unc"]:.3f})')

    # Plot min and max points with error bars
    ax.errorbar(result['min_pos'], result['min_value'],
                yerr=result['min_unc'],
                fmt='v', color='red', markersize=12,
                capsize=5, capthick=2, linewidth=2,
                label=f'Min: {result["min_value"]:.3f}', zorder=5)

    ax.errorbar(result['max_pos'], result['max_value'],
                yerr=result['max_unc'],
                fmt='^', color='green', markersize=12,
                capsize=5, capthick=2, linewidth=2,
                label=f'Max: {result["max_value"]:.3f}', zorder=5)

    # Draw displacement arrow
    arrow_x = (result['min_pos'] + result['max_pos']) / 2
    ax.annotate('',
                xy=(arrow_x, result['max_value']),
                xytext=(arrow_x, result['min_value']),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=2.5, shrinkA=0, shrinkB=0))

    # Add displacement text
    ax.text(arrow_x + (dist.max() - dist.min()) * 0.02,
            (result['min_value'] + result['max_value']) / 2,
            f"Δ = {result['displacement']:.3f}\n± {result['uncertainty']:.3f}",
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='black', alpha=0.9))

    # Styling
    ax.set_xlabel('Distance along profile', fontweight='bold', fontsize=12)
    ax.set_ylabel('Displacement', fontweight='bold', fontsize=12)
    ax.set_title(f'Displacement Analysis with Min/Max Windows, frame #{frame_n}',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig, ax


def plot_displacement_analysis(dist, profile, result, frame_n, profile_smooth=None):
    """
    Plot profile with extrema and window regions.
    Now supports visualization of linear regression fits.

    Parameters:
    -----------
    dist: distance array
    profile: profile values
    result: dictionary output from compute_displacement_from_extrema
    frame_n: frame number for title
    profile_smooth: optional smoothed profile
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot original profile
    ax.plot(dist, profile, 'o-', color='gray', alpha=0.6,
            markersize=3, linewidth=1, label='Profile')

    # Plot smoothed profile if available
    if profile_smooth is not None:
        ax.plot(dist, profile_smooth, 'k--', linewidth=2,
                alpha=0.7, label='Smoothed')

    # Shaded bands for min and max windows
    y_min, y_max = ax.get_ylim()

    ax.fill_between(dist, y_min, y_max,
                    where=result['min_window_mask'],
                    alpha=0.3, color='red',
                    label=f'Min window (±{result["min_unc"]:.3f})')

    ax.fill_between(dist, y_min, y_max,
                    where=result['max_window_mask'],
                    alpha=0.3, color='green',
                    label=f'Max window (±{result["max_unc"]:.3f})')

    # If linear method, plot the fitted lines
    if result.get('method') == 'linear':
        # Plot min window linear fit
        if result.get('min_slope') is not None:
            min_dist_fit = dist[result['min_window_mask']]
            min_fit_vals = result['min_slope'] * min_dist_fit + result['min_intercept']
            ax.plot(min_dist_fit, min_fit_vals, 'r-', linewidth=3,
                    label='Min linear fit', zorder=4)

        # Plot max window linear fit
        if result.get('max_slope') is not None:
            max_dist_fit = dist[result['max_window_mask']]
            max_fit_vals = result['max_slope'] * max_dist_fit + result['max_intercept']
            ax.plot(max_dist_fit, max_fit_vals, 'g-', linewidth=3,
                    label='Max linear fit', zorder=4)

        # Optional: extrapolate lines to show offset at midpoint
        midpoint = (result['min_pos'] + result['max_pos']) / 2
        if result.get('min_slope') is not None and result.get('max_slope') is not None:
            # Extend min line
            extend_min_x = np.array([result['min_pos'], midpoint])
            extend_min_y = result['min_slope'] * extend_min_x + result['min_intercept']
            ax.plot(extend_min_x, extend_min_y, 'r--', linewidth=1.5, alpha=0.5, zorder=3)

            # Extend max line
            extend_max_x = np.array([midpoint, result['max_pos']])
            extend_max_y = result['max_slope'] * extend_max_x + result['max_intercept']
            ax.plot(extend_max_x, extend_max_y, 'g--', linewidth=1.5, alpha=0.5, zorder=3)

    # Plot min and max points with error bars
    ax.errorbar(result['min_pos'], result['min_value'],
                yerr=result['min_unc'],
                fmt='v', color='red', markersize=12,
                capsize=5, capthick=2, linewidth=2,
                label=f'Min: {result["min_value"]:.3f}', zorder=5)

    ax.errorbar(result['max_pos'], result['max_value'],
                yerr=result['max_unc'],
                fmt='^', color='green', markersize=12,
                capsize=5, capthick=2, linewidth=2,
                label=f'Max: {result["max_value"]:.3f}', zorder=5)

    # Draw displacement arrow
    arrow_x = (result['min_pos'] + result['max_pos']) / 2
    ax.annotate('',
                xy=(arrow_x, result['max_value']),
                xytext=(arrow_x, result['min_value']),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=2.5, shrinkA=0, shrinkB=0))

    # Add displacement text
    method_label = result.get('method', 'mean')
    ax.text(arrow_x + (dist.max() - dist.min()) * 0.02,
            (result['min_value'] + result['max_value']) / 2,
            f"Δ = {result['displacement']:.3f}\n± {result['uncertainty']:.3f}\n({method_label})",
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='black', alpha=0.9))

    # Styling
    ax.set_xlabel('Distance along profile', fontweight='bold', fontsize=12)
    ax.set_ylabel('Displacement', fontweight='bold', fontsize=12)
    ax.set_title(f'Displacement Analysis with Min/Max Windows, frame #{frame_n}',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig, ax


def surface_to_los(d_east, d_north, d_up, incidence_deg, heading_deg):
    """
    Project 3D surface displacement to InSAR LOS.

    Parameters:
    -----------
    d_east, d_north, d_up: displacement components (same units)
    incidence_deg: satellite incidence angle (degrees, typically 30-45°)
    heading_deg: satellite heading angle (degrees)
                 Ascending: ~-13° (flying S to N, looking E)
                 Descending: ~-167° (flying N to S, looking W)

    Returns:
    --------
    d_los: displacement in LOS direction
    """
    # Convert to radians
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)

    # LOS projection
    d_los = (d_east * np.sin(theta) * np.cos(alpha) -
             d_north * np.sin(theta) * np.sin(alpha) +
             d_up * np.cos(theta))
    s_e = -1. * np.sin(alpha) * np.sin(theta)
    s_n = -1. * np.cos(alpha) * np.sin(theta)
    s_u = np.cos(theta)
    return d_los


def plot_transects_new(data, pred, lon, lat, point_lon, point_lat, np_creep_time, creep_data, creep_mask, datetimes):
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
    norm = TwoSlopeNorm(vmin=-2., vcenter=0, vmax=1.)

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
    cbar.set_ticklabels([str(int(norm.vmin)), str(norm.vcenter), str(int(norm.vmax))])

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

        # d0_fit, d0_std, popt = compute_static_displacement_model(dd_i, pp_i)
        # d0_fit, x0_fit, w_fit, offset_fit = popt
        '''plt.plot(dd_i, pp_i)
        plt.plot(dd_i, arctangent_model(dd_i, *popt))
        plt.title(d0_fit)
        plt.show()'''

        # acc_d.append(np.abs(d0_fit))
        # acc_d_std.append(d0_std)

        # plot_displacement_analysis(dd_i, pp_i, res, i + 1, profile_smooth=gaussian_filter1d(pp_i, sigma=smoothing_sigma))
        # plt.show()

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

    # ax_prof.legend(loc='best', fontsize=8)
    # fig.tight_layout()
    plt.savefig('./ismetpasa_transect.pdf',
                bbox_inches='tight')


if __name__ == '__main__':
    ismetpasa_strike = 81.5  # np.rad2deg(np.arctan2(0.625, 0.09375))
    creep_obliquity = 33  # degrees
    rawts, topo, lon, lat, velocity, datetimes, nan_mask = load_ts()
    print(velocity.shape, lat.shape, lon.shape, rawts.shape)
    np_creep_time, creep_data = load_creepmeter_data()
    ismetpasa_lat, ismetpasa_lon = 40.8698, 32.6258

    # creep_data = creep_data / np.cos(np.deg2rad(creep_obliquity))  # obtain the fault-parallel, but not necessary for North Wall

    start_creep = np.datetime64('2015-05-27')
    end_creep = np.datetime64('2018-04-11')

    creep_mask = (np_creep_time >= start_creep) & (np_creep_time <= end_creep)
    selected_creep = creep_data[creep_mask]
    print('Total accumulated creep:', selected_creep[-1] - selected_creep[0])

    # crop idx
    # x0, x1 = 411, 539
    x0, x1 = 420, 548  # 411 -1, 539 - 1
    y0, y1 = 51, 179  # 34 - 4, 162 - 4
    t0, t1 = 1, 10
    # crop ts
    data = rawts[t0:t1, y0:y1, x0:x1]
    datetimes = datetimes[t0:t1]
    topo = topo[y0:y1, x0:x1]
    nan_mask = nan_mask[t0:t1, y0:y1, x0:x1]
    lon_crop = lon[y0:y1, x0:x1]
    lat_crop = lat[y0:y1, x0:x1]

    # renormalize topography after crop
    topo = (topo - np.nanmin(topo)) / (np.nanmax(topo) - np.nanmin(topo))

    data, topo = data[None, ..., None], topo[None, None, ..., None]

    pred = predict(denoiser_configuration, data, data, topo)

    data, pred = data.squeeze(), pred.squeeze()

    data[nan_mask] = np.nan
    pred[nan_mask] = np.nan

    '''make_plot(velocity, lat, lon, data, pred, x0, x1, y0, y1, ismetpasa_lat, ismetpasa_lon, np_creep_time, creep_data,
              nan_mask, lon_crop, lat_crop, datetimes)'''

    '''make_plot_3rows(velocity, lat, lon, data, pred, x0, x1, y0, y1, ismetpasa_lat, ismetpasa_lon, np_creep_time, creep_data,
              nan_mask, lon_crop, lat_crop, datetimes)'''

    # plot_transects(pred[-1])

    plot_transects_new(data, pred, lon_crop, lat_crop, ismetpasa_lon, ismetpasa_lat, np_creep_time, creep_data,
                       creep_mask, datetimes)
