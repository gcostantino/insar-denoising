"""
Author: Giuseppe Costantino, February 2025.
I decided to split the parameters into user-accessible and private ones. Please do not modify the latter ones,
as they are associated with structural properties of the methods and should not be edited by users.
"""

N = 128  # square window is considered. This code could be extended to rectangular windows in the future
Nt = 9  # number of frames in the InSAR time series
batch_size = 2 ** 10  # number of samples to generate in a single batch (for a single 'generation')
n_faults = 8  # num. of dislocation models that are actually created (fault model reused batch_size // n_faults times)
mogi_proba = 0.5  # insert a Mogi source with a given probability
spatial_window_scale = 111.  # we create a fake-UTM space (~1 degree)

# ***************************************** satellite-related parameters  *********************************************#
heading_angle_range = (0, 360)  # degrees (exploit a full configuration of SAR satellites)
incidence_angle_range = (35, 40)  # degrees
# **********************************************************************************************************************#
noise_amplitude_range = (
0., 10.)  # centimeters  # (10., 500.)  # mm. General range for InSAR phase noise. Assumed symmetric here
bilateral_noise_amplitude_range = (-10., 10.)  # centimeters. If you want to tune just refer to the abs(), e.g., 10 cm
# **************************************** parameters for physical models  ********************************************#
depth_range = (0, 20)  # km
fault_strike_range = (0, 360)  # degrees
fault_dip_range = (30, 90)  # degrees
fault_rake_range = (-180, 180)  # degrees
fault_length_range = (50, 250)  # km
fault_width_range = (5, 20)  # km
fault_slip_range = (2., 500.)  # centimeters # (20, 5000)  # mm
fault_slip_onset_time_range = (-4, 13)  # frame-wise. values outside (0, Nt-1) mean that the event starts/ends off-frame
fault_slip_duration_sigma_range = (0.1, 4.)  # see (1)

mogi_source_depth_range = (300, 2e3)  # m
mogi_source_volume_change_range = (5e4, 5e5)  # m^3
mogi_source_amplitude_lim = (1., 100.)  # centimeters (~ to fault-induced displ.). NW: this is surface displacement!
mogi_onset_time_range = (0, 9)  # frame-wise
mogi_duration_sigma_range = (0.01, 1)  # see (1)

# (1) Giving the time duration as a function of the standard deviation of a Gaussian CDF is quite obscure. Here below
# I built a table with approximate sigma-duration relationships (the code accepts only sigma values, but it would make
# more sense to think at the duration as a time quantity, right? One could ask why I didn't rewrite the code to replace
# sigma with an actual duration. I wanted. But I am lazy.):

# +----------+-----------------+-----------------+-------------------+
# | sigma(σ) | 68% range (±1σ) | 95% range (±2σ) | 99.7% range (±3σ) |
# +----------+-----------------+-----------------+-------------------+
# | 0.5      | 1 frame         | ~2 frames       | ~3 frames         |
# +----------+-----------------+-----------------+-------------------+
# | 1.0      | 2 frames        | ~4 frames       | ~6 frames         |
# +----------+-----------------+-----------------+-------------------+
# | 1.5      | 3 frames        | ~6 frames       | ~9 frames         |
# +----------+-----------------+-----------------+-------------------+
# | 2.0      | 4 frames        | ~8 frames       | ~12 frames        |
# +----------+-----------------+-----------------+-------------------+
# | 3.0      | 6 frames        | ~12 frames      | ~12 frames        |
# +----------+-----------------+-----------------+-------------------+
# 
# use the ±2σ range to find the corresponding duration (e.g., for σ=1, most of the events happens within ~4 time steps.
# *********************************************************************************************************************#

########################################################################################################################
########################################################################################################################
########################################################################################################################
###########################################  private, structural parameters ############################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
fault_proba = 0.9  # allow for pure-noise samples 10% of the time, improve generalization of the denoiser (*)
secondary_fault_proba = 0.5  # change that to generate diverse training samples (e.g., increasing difficulty)
transient_tertiary_fault_proba = 0.5
dt = 1  # time spacing between frames
lat_lon_moving_ratio_inside_window = 0.95  # generate dislocation epicenters inside 80% of original box
fault_slip_duration_sigma_min_inside_win = .1
fault_slip_duration_sigma_min_outside_win = 2.
mogi_spatial_scale = 2.  # unitless
nu = 0.25  # poisson's ratio for the medium

sigma_roughness_exponent_range = (0, 3)
hurst_coefficient_range = (.5, 3.)

topo_noise_autocorr_range = (0.9, 0.999)  # auto-correlation of topography-dependent noise
topo_autoregressive_model_mean = 0.
topo_autoregressive_model_std_max = 1.

atmo_noise_sigma_range = (0.1, 1.)  # std of atmospheric noise amplitude
atmo_noise_corr_length_range = (5, 1000)  # num. of pixels

n_lambda_turbulent_delay = 100  # number of characteristic wavelengths for turbulent multiscale noise
correlated_pixels_range = (3, 1000)  # min/max number of correlated pixel in turbulent atmospheric delay model
beta_distribution_alpha = 1.  # do NOT change
beta_distribution_beta = 3.5  # do NOT change

stratified_lin_coef_range = (0., 10.)  # 0 times to 10 times RMS
stratified_qua_coef_range = (0., 10.)
turbulent_coef_range = (0., 10.)

rand_noise_sigma = 10 / 3  # stf of Normal(0,sigma**2) for constant offsets (~ between -10 and 10 cm)

rand_noise_amplitude_range = (-5, 5)  # random noise. No unit of measure, just obscure stuff. DO NOT modify.

glitch_frac_exp_lim = (-4, -1)  # unitless. Fraction of glitchy pixels
glitchy_pixel_amplitude_max = 10.  # centimeters # 100.  # mm

buggy_patches_frac_max = 0.1  # % of buggy patches in the data set
buggy_noise_corr = 0.995  # auto-correlation of an autoregressive model used to model fluctuations inside buggy patches
buggy_patch_corr = 0.999  # auto-correlation of an autoregressive model used to model fluctuations inside buggy patches
buggy_patches_amplitude_max = 10.  # centimeters # 100.  # mm

decorrelation_pixel_value_range = (-10., 10.)  # centimeters

buggy_surf_mask_min_thresh = 0.5  # threshold for artificial surface that models the buggy patches

noise_scale_coeff = 1.0  # legacy parameter (pretty much useless). However, DO NOT modify.
mean_noise_scale = -3.  # legacy parameter (pretty much useless). However, DO NOT modify.
sigma_noise_scale = 1.  # legacy parameter (pretty much useless). However, DO NOT modify.

surface_displacement_range = None  # centimeters  # Used to rescale surface displacements (used for intermediate steps of curriculum learning)
minimum_los_surface_displacement = 0.1  # centimeters
normalize_temporal_slip_evolution = True  # used to normalize slip evolution between 0 and 1

# (*) this parameter should not be changed. A 10-15% exposure to learn the noise structure is enough, in order to avoid
# overwhelming the model with too much 'low-SNR-like' signals (otherwise the model will be biased towards
# zero-displacement outputs)

# *********************************************************************************************************************#
# On the unit measure of fault length, width, depth etc...
# These bounds are physically meaningless and the length parameters do not have any intrinsic unit measure, since
# everything depends on the pixel size, which, at test phase, is variable since it depend on the specific satellite,
# postprocessing, multi-looking, etc...
# However, here the box is assumed to have 1x1 degrees^2 units, thus the length should be thought as % of the total box,
# which makes way more sense [at least for me].
