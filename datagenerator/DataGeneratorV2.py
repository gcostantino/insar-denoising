"""
Author: Giuseppe Costantino, February 2025.
"""
import os

import numpy as np
import scipy.fft as fftw
import scipy.spatial.distance as scidis
from scipy.ndimage import rotate

from okada.okada import forward as okada85
from utils.modeldeform_utils import mogi_defo
from utils.stats_utils import gaussian_cdf


class DataGeneratorV2:
    def __init__(self, param_dict, dtype=np.float32, verbose=False, seed=None, batch_idx=None,
                 precomputed_turbulent_maps=False):
        """Set up sizes
        * Nt: number of time frame
        * N : number of pixels in X and Y (square window)"""

        self.los = None
        self.Nt = param_dict['Nt']
        self.N = param_dict['N']
        self.verbose = verbose
        self.dtype = dtype
        self.batch_idx = batch_idx
        self.precomputed_turbulent_maps = precomputed_turbulent_maps

        self.fault_rotation_angles = [0, 90, 180, 270]
        self.spatial_window_scale = param_dict['spatial_window_scale']

        self.xs = np.linspace(-self.spatial_window_scale / 2., self.spatial_window_scale / 2., self.N,
                              dtype=self.dtype)
        self.ys = np.linspace(-self.spatial_window_scale / 2., self.spatial_window_scale / 2., self.N,
                              dtype=self.dtype)
        self.Xs, self.Ys = np.meshgrid(self.xs, self.ys, indexing='xy')
        self.Zs = np.zeros(self.Xs.shape, dtype=self.dtype)
        self.Xs, self.Ys, self.Zs = self.Xs.flatten(), self.Ys.flatten(), self.Zs.flatten()

        self.x_noise, self.y_noise = np.meshgrid(np.linspace(1, self.N, self.N, dtype=self.dtype),
                                                 np.linspace(1, self.N, self.N, dtype=self.dtype))
        self.distances = np.min(
            scidis.cdist(np.stack((self.x_noise.flatten(), self.y_noise.flatten()), dtype=self.dtype).T,
                         np.array([[1, 1], [1, self.N], [self.N, self.N],
                                   [self.N, 1]], dtype=self.dtype)).reshape(
                (self.N, self.N, 4)), axis=2)

        self.seed = seed if seed is not None else np.random.SeedSequence().generate_state(1)[0]
        self.rng = np.random.default_rng(self.seed)  # Independent RNG instance
        self.parameters = param_dict

    def build_los(self, head=None, inc=None):
        """Create LOS vector (unique) from incidence and heading, which
        can either be fixed or randomly sampled (ENU) """

        if head is None:
            head = self.rng.uniform(*self.parameters['heading_angle_range'])
        if inc is None:
            inc = self.rng.uniform(*self.parameters['incidence_angle_range'])
        if self.verbose:
            print("LOS heading and incidence")
            print(head, inc)

        alpha = (head + 90.) * np.pi / 180.
        phi = inc * np.pi / 180.

        s_e = -1. * np.sin(alpha) * np.sin(phi)
        s_n = -1. * np.cos(alpha) * np.sin(phi)
        s_u = np.cos(phi)

        self.los = np.array([s_e, s_n, s_u], dtype=self.dtype)

    def artificial_self_affine_surface(self, rms_height, hurst_exponent,
                                       domain_length_x=1, domain_length_y=1,
                                       short_wavelength_cutoff=None, long_wavelength_cutoff=None,
                                       rms_slope=None, rolloff=1.0):
        original_nx, original_ny = self.N, self.N
        padded_nx = int(original_nx * 1.5)
        padded_ny = int(original_ny * 1.5)
        crop_offset = (padded_nx - original_nx) // 2

        # Define spectral cutoffs
        q_max = 2 * np.pi / short_wavelength_cutoff if short_wavelength_cutoff else np.pi * min(
            padded_nx / domain_length_x, padded_ny / domain_length_y)
        q_min = 2 * np.pi / long_wavelength_cutoff if long_wavelength_cutoff else 2 * np.pi * max(1 / domain_length_x,
                                                                                                  1 / domain_length_y)

        area = domain_length_x * domain_length_y

        # Scaling factor to match desired RMS height or slope
        if rms_height is not None:
            factor = (2 * rms_height /
                      np.sqrt(q_min ** (-2 * hurst_exponent) - q_max ** (-2 * hurst_exponent)) *
                      np.sqrt(hurst_exponent * np.pi))
        elif rms_slope is not None:
            factor = (2 * rms_slope /
                      np.sqrt(q_max ** (2 - 2 * hurst_exponent) - q_min ** (2 - 2 * hurst_exponent)) *
                      np.sqrt((1 - hurst_exponent) * np.pi))
        else:
            raise ValueError('You must specify either rms_height or rms_slope.')

        C0 = factor * padded_nx * padded_ny / np.sqrt(area)

        # Initialize arrays
        num_cols_fft = padded_ny // 2 + 1
        real_space_surface = np.zeros((padded_nx, padded_ny), dtype=self.dtype)
        fourier_coefficients = np.zeros((padded_nx, num_cols_fft), dtype=np.complex128)

        # Prepare wavevector magnitudes
        wavevector_y = 2 * np.pi * np.arange(num_cols_fft) / domain_length_y

        for row in range(padded_nx):
            wavevector_x = (2 * np.pi * (padded_nx - row) / domain_length_x
                            if row > padded_nx // 2
                            else 2 * np.pi * row / domain_length_x)

            q_squared = wavevector_x ** 2 + wavevector_y ** 2
            if row == 0:
                q_squared[0] = 1.0  # avoid division by zero

            # Generate random phase and Gaussian-distributed amplitude
            random_phase = np.exp(2j * np.pi * self.rng.uniform(size=num_cols_fft))
            gaussian_amplitude = self.rng.normal(size=num_cols_fft)
            spectrum = C0 * random_phase * gaussian_amplitude

            # Apply power-law scaling
            fourier_coefficients[row, :] = spectrum * q_squared ** (-(1 + hurst_exponent) / 2)

            # Apply cutoffs
            fourier_coefficients[row, q_squared > q_max ** 2] = 0.0
            low_q_mask = q_squared < q_min ** 2
            fourier_coefficients[row, low_q_mask] = (
                    rolloff * spectrum[low_q_mask] * q_min ** (-(1 + hurst_exponent))
            )

        # Enforce Hermitian symmetry
        if padded_nx % 2 == 0:
            fourier_coefficients[0, 0] = np.real(fourier_coefficients[0, 0])
            fourier_coefficients[1:padded_nx // 2, 0] = (
                fourier_coefficients[-1:padded_nx // 2:-1, 0].conj()
            )
        else:
            fourier_coefficients[0, 0] = np.real(fourier_coefficients[0, 0])
            fourier_coefficients[padded_nx // 2, 0] = np.real(fourier_coefficients[padded_nx // 2, 0])
            fourier_coefficients[1:padded_nx // 2, 0] = (
                fourier_coefficients[-1:padded_nx // 2 + 1:-1, 0].conj()
            )

        # Inverse FFT to real space
        for col in range(num_cols_fft):
            fourier_coefficients[:, col] = np.fft.ifft(fourier_coefficients[:, col])
        for row in range(padded_nx):
            real_space_surface[row, :] = np.fft.irfft(fourier_coefficients[row, :])

        # Crop back to original size
        cropped_surface = real_space_surface[crop_offset:crop_offset + original_nx,
                          crop_offset:crop_offset + original_ny]
        return cropped_surface

    def autoregressive_order_one_process(self, corr, mu=0., sigma=1.):
        """topo related noise in time
        * corr : temporal auto-correlation, 0.999 gives linear-like trends"""

        assert 0 < corr < 1, "Auto-correlation must be between 0 and 1"
        c = mu * (1 - corr)
        sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

        # Sample the auto-regressive process.
        signal = [c + self.rng.normal(0, sigma_e)]
        for _ in range(1, self.Nt):
            signal.append(c + corr * signal[-1] + self.rng.normal(0, sigma_e))

        return np.array(signal, dtype=self.dtype)

    def random_gaussian_time_evolution(self):
        return self.rng.normal(loc=1., scale=1., size=self.Nt).astype(self.dtype)

    def single_wavelength_turbulent_noise(self, sig=None, lam=None):
        """map of atmo related noise as white spatially correlated noise
        * sig : standard deviation of noise amplitude
        * lam : spatial length of correlation in pixel number
        """

        if sig is None:
            sig = self.rng.uniform(*self.parameters['atmo_noise_sigma_range'])
        if lam is None:
            lam = self.rng.uniform(*self.parameters['atmo_noise_corr_length_range'])

        if self.verbose:
            print("Atmo noise: sigma {}, correlation length {}".format(sig, lam))

        # Generate white noise

        white = self.rng.random(size=(self.N, self.N), dtype=self.dtype)
        correl = sig * np.exp(-self.distances / lam)

        # FFT it
        fwhite = fftw.fft2(white)
        fcorrel = fftw.fft2(correl)

        noise = np.real(fftw.ifft2(fwhite * fcorrel))
        noise -= np.mean(noise)

        return noise

    def turbulent_multiscale_noise(self):
        Nx, Ny = self.N, self.N
        n_lambda = self.parameters['n_lambda_turbulent_delay']
        alpha, beta = self.parameters['beta_distribution_alpha'], self.parameters['beta_distribution_beta']
        total_noise = np.zeros((Nx, Ny))
        min_lam_pix, max_lam_pix = self.parameters['correlated_pixels_range']
        lambda_list = min_lam_pix + (max_lam_pix - min_lam_pix) * self.rng.beta(alpha, beta, size=n_lambda)
        sigma_list = self.rng.uniform(*self.parameters['atmo_noise_sigma_range'], size=n_lambda)
        for idx in range(n_lambda):
            sigma, lamb = sigma_list[idx], lambda_list[idx]
            sigma = sigma / lamb  # normalize power to have true sigma**2 variance at the end
            nc = self.single_wavelength_turbulent_noise(sigma, lamb)
            total_noise += nc
        return total_noise

    def buggy_noise(self, size_x, size_y, sigma=1.):
        """patch of noise on subarea?"""

        lam = self.rng.uniform(1, 3)  # 30.
        # topo_coupling=self.rng.uniform(3.0,10.0)

        # Generate white noise
        white = self.rng.random(size=(size_x, size_y), dtype=self.dtype)

        # Generate correl
        x, y = np.meshgrid(np.linspace(1, size_x, size_x, dtype=self.dtype),
                           np.linspace(1, size_y, size_y, dtype=self.dtype))
        distances = np.min(scidis.cdist(np.stack((x.flatten(), y.flatten()), dtype=self.dtype).T,
                                        np.array([[1, 1], [1, size_x], [size_x, size_y], [size_y, 1]],
                                                 dtype=self.dtype)).reshape((size_x, size_y, 4), order='F'), axis=2)
        correl = sigma * np.exp(-distances / lam)

        # FFT it
        fwhite = fftw.fft2(white)
        fcorrel = fftw.fft2(correl)

        noise = np.real(fftw.ifft2(fwhite * fcorrel))
        noise -= np.mean(noise)

        return noise

    def defo_history(self, mu=None, sig=None, mu_lim=None, sig_lim=None, adjust_sigma_wrt_mu=True):
        """Returns: time series of slip (length = 9).
        New in this version: sigma (i.e., event duration) depends on mu. Slip events can begin/end out of the
        observation window. Sigma is adjusted as a function of mu such that ~.5mm slip is visible.
        mu limits: window +/- window/2."""
        assert mu_lim is not None
        assert sig_lim is not None
        if mu is None:
            mu = self.rng.uniform(*mu_lim)
        # after choosing mu, define sigma limits, and choose sigma
        #############
        if adjust_sigma_wrt_mu:
            if 0 <= mu < 9:  # this can be replaced by more sophisticated selection criteria
                # also, this currently works only for dislocations, not for Mogi sources
                sigma_min = self.parameters['fault_slip_duration_sigma_min_inside_win']
            else:
                sigma_min = self.parameters['fault_slip_duration_sigma_min_outside_win']

        # Choose sigma within the new limits
        if sig is None:
            if adjust_sigma_wrt_mu:
                sig = self.rng.uniform(max(sigma_min, sig_lim[0]), sig_lim[1])
            else:
                sig = self.rng.uniform(*sig_lim)

        tspace = np.arange(self.Nt, dtype=self.dtype) * self.parameters['dt']
        if self.Nt > 1:
            idx = self.rng.integers(1, self.Nt)  # add irregularity in tsampling
            tspace[idx] += self.rng.choice([-0.5, 0.5]) * self.parameters['dt']

        gaussian_rate = np.array([gaussian_cdf(val, mu, sig) for val in tspace], dtype=self.dtype)
        time_deformation = gaussian_rate

        if self.verbose:
            print("time of event {}, temporal width {}".format(mu, sig))

        return time_deformation

    def model_fault_deformation(self):
        """create fault deformation
        """
        # fault_deformation = np.zeros((self.Nt, self.N, self.N, 1))  # Time, lat, lon, 1

        # the fault centroid can move freely in an inner box of 80% original dimensions
        delta_latlon = self.parameters['lat_lon_moving_ratio_inside_window']
        d_lat = self.rng.uniform(-delta_latlon, delta_latlon) * self.spatial_window_scale / 2
        d_lon = self.rng.uniform(-delta_latlon, delta_latlon) * self.spatial_window_scale / 2

        lat, lon = d_lat, d_lon

        depth = self.rng.uniform(*self.parameters['depth_range'])
        strike_a = self.rng.integers(*self.parameters['fault_strike_range'])
        dip_a = self.rng.integers(*self.parameters['fault_dip_range'])
        rake_a = self.rng.integers(*self.parameters['fault_rake_range'])
        length = self.rng.uniform(*self.parameters['fault_length_range'])
        width = self.rng.uniform(*self.parameters['fault_width_range'])

        if self.parameters['minimum_los_surface_displacement'] is not None:
            # in this case, unit slip is constrained by the other parameters
            # assign unit slip to compute the predicted surface displacement to unit slip.
            # based on the threshold surface displacement, then we randomly draw slip in a legit interval (see below)
            # by enforcing the linearity between surface displacement and average fault slip at depth
            fault_slip = 1.
        else:
            fault_slip = self.rng.uniform(*self.parameters['fault_slip_range'])

        if self.verbose:
            print("Compute fault deformation")
            print("lat,lon,depth: {:.2f}, {:.2f}, {:.2f}".format(lat, lon, depth))
            print("strike {:.2f},dip {:.2f},length {:.2f},width {:.2f}".format(strike_a, dip_a, length, width))

        temporal_slip_evolution = self.defo_history(mu_lim=self.parameters['fault_slip_onset_time_range'],
                                                    sig_lim=self.parameters['fault_slip_duration_sigma_range'])
        if self.parameters['normalize_temporal_slip_evolution']:
            min_ev, max_ev = np.min(temporal_slip_evolution), np.max(temporal_slip_evolution)
            temporal_slip_evolution = (temporal_slip_evolution - min_ev) / (max_ev - min_ev)

        temporal_slip_evolution *= self.rng.choice([-1.0, 1.0])

        depth_center = depth + np.sin(dip_a * np.pi / 180.) * width / 2.
        # instead of using a for loop in the time dimension, I exploit the linearity of Okada equations wrt the slip,
        # under the assumption of no tensile slip component. Thus, the predicted deformation can be computed once for
        # unit slip and then modulated by the temporal evolution.
        defo_enu = okada85(self.Xs, self.Ys, xoff=lon, yoff=lat, depth=depth_center, length=length, width=width,
                           slip=fault_slip, opening=0., strike=strike_a, dip=dip_a, rake=rake_a)

        defo_enu = np.array(defo_enu, dtype=self.dtype).T
        defo_enu[np.isnan(defo_enu)] = 0.  # check for singularities...
        defo_los = np.dot(defo_enu, self.los)

        if self.parameters['minimum_los_surface_displacement'] is not None:
            # here, defo_los is the LOS deformation associated to unit fault slip.
            max_defo = np.max(np.abs(defo_los))
            min_fault_slip = self.parameters['minimum_los_surface_displacement'] / max_defo
            max_fault_slip = self.parameters['fault_slip_range'][1]
            admissible_fault_slip_range = (min_fault_slip, max_fault_slip)
            if min_fault_slip >= max_fault_slip:
                raise ValueError(f"{admissible_fault_slip_range} is not a valid slip range.")
            admissible_fault_slip = self.rng.uniform(*admissible_fault_slip_range)
            defo_los = defo_los * admissible_fault_slip

        los_deformation = defo_los.reshape((self.N, self.N))
        fault_deformation = (temporal_slip_evolution[:, None, None] * los_deformation)[..., None]
        if self.parameters['surface_displacement_range'] is not None:
            # rescale fault deformation with affine transformation: (def - def_min)/(def_max - def_min) * (u - c) + c
            # with u ~ Uniform(min_def_range, max_def_range), c = min def.
            global_min, global_max = fault_deformation.min(), fault_deformation.max()
            if global_min * global_max > 0:  # same sign
                m = (global_max + global_min) / 2
                fault_deformation = fault_deformation - m
                global_min, global_max = global_min - m, global_max - m
            rand_range = self.rng.uniform(*self.parameters['surface_displacement_range'])
            new_min = rand_range * global_min / (global_max - global_min)  # proportional to old aspect ratio
            scaled_los_fault_defo = (fault_deformation - global_min) / (global_max - global_min)
            rescaled_fault_deformation = scaled_los_fault_defo * rand_range + new_min
            return rescaled_fault_deformation, fault_deformation
        return fault_deformation, fault_deformation  # a bit ugly I have to admit...

    def _model_mogi_source(self):
        ''' create mogi deformation
        * geombounds : bounds for mogi geometry list of tuples in m or m3 ?
                [(min_depth,max_depth),(min_dV...)]
                initially : [(300,1500),(5E4,5E5)]
        * timebounds : bounds for time of occurence and duration
                [(min_t, max_t),(min_wt, max_wt)]
                initially : [(0,7), (0.33,0.55)]'''

        # Set parameters
        scale = self.parameters['mogi_spatial_scale']
        xcen = self.rng.uniform(-scale, scale) * 1e3  # in m
        ycen = self.rng.uniform(-scale, scale) * 1e3
        d = self.rng.uniform(*self.parameters['mogi_source_depth_range'])
        dV = self.rng.uniform(*self.parameters['mogi_source_volume_change_range']) * self.rng.choice([-1, 1])

        # frame
        x = np.linspace(-scale, scale, self.N, dtype=self.dtype) * 1e3
        y = np.linspace(-scale, scale, self.N, dtype=self.dtype) * 1e3
        X, Y = np.meshgrid(x, y)

        # Run mogi model with delta volume input
        deformation = self.defo_history(mu_lim=self.parameters['mogi_onset_time_range'],
                                        sig_lim=self.parameters['mogi_duration_sigma_range'])
        los_defo = []
        for t in range(len(deformation)):
            defo = mogi_defo(X, Y, xcen, ycen, d, dV * deformation[t], self.parameters['nu'])
            los_defo.append(np.dot(defo.swapaxes(0, -1), self.los))

        return np.array(los_defo, dtype=self.dtype).reshape((self.Nt, self.N, self.N, 1))

    def model_volcano_deformation(self, scale_defo=True):
        mogi_scale = self.rng.uniform(*self.parameters['mogi_source_amplitude_lim'])
        mogi_source = self._model_mogi_source()
        if scale_defo:
            max_mogi = np.abs(mogi_source).max()
            norm_mogi_source = mogi_source / max_mogi if max_mogi != 0 else mogi_source
            scaled_mogi_source = norm_mogi_source * mogi_scale
            return scaled_mogi_source
        return mogi_source

    def compute_turbulent_map(self, idx=None):
        if self.precomputed_turbulent_maps:
            return np.load(os.path.join(self.parameters['turbulent_map_path'],
                                        self.batch_idx * self.parameters['batch_size'] + idx + '.npz'))
        else:
            return np.array([self.turbulent_multiscale_noise() for _ in range(self.Nt)])

    def model_noise_contributions(self, gen_idx, return_noise_components=False):
        topo = self.artificial_self_affine_surface(
            10. ** (-self.rng.integers(*self.parameters['sigma_roughness_exponent_range'])),
            self.rng.uniform(*self.parameters['hurst_coefficient_range']))
        autocorr = self.rng.uniform(*self.parameters['topo_noise_autocorr_range'])
        sigma = self.rng.uniform(high=self.parameters['topo_autoregressive_model_std_max'])

        # create stratified atmospheric delay, topography-dependent noise contribution
        '''topo_t_evolution = self.autoregressive_order_one_process(autocorr, self.parameters[
            'topo_autoregressive_model_mean'], sigma)'''
        topo_t_evolution = self.random_gaussian_time_evolution()

        normalized_topo = 2 * (topo - np.min(topo)) / (np.max(topo) - np.min(topo)) - 1

        topo_lin = np.array([topo_t_evolution[k] * normalized_topo for k in range(self.Nt)])
        topo_qua = np.array([topo_t_evolution[k] * normalized_topo ** 2 for k in range(self.Nt)])
        # turb_noise = np.array([self.turbulent_multiscale_noise() for _ in range(self.Nt)])
        turb_noise = self.compute_turbulent_map(gen_idx)

        # normalize/re-expand the noise contributions
        norm_topo_lin = 2 * ((topo_lin - np.min(topo_lin)) / (np.max(topo_lin) - np.min(topo_lin))) - 1
        norm_topo_qua = 2 * ((topo_qua - np.min(topo_qua)) / (np.max(topo_qua) - np.min(topo_qua))) - 1
        norm_turbulent = 2 * (turb_noise - np.min(turb_noise)) / (np.max(turb_noise) - np.min(turb_noise)) - 1

        # Combine stratified and turbulent delays via a, b, c coefficients

        a = self.rng.uniform(*self.parameters['stratified_lin_coef_range'])
        b = self.rng.uniform(*self.parameters['stratified_qua_coef_range'])
        c = self.rng.uniform(*self.parameters['turbulent_coef_range'])

        sum_noise = a * norm_topo_lin + b * norm_topo_qua + c * norm_turbulent

        # now we rescale this to real units (centimeters) via an affine transformation
        # we first normalize
        norm_sum = 2 * ((sum_noise - np.min(sum_noise)) / (np.max(sum_noise) - np.min(sum_noise))) - 1

        # then prepare for the affine transformation
        min_noise, max_noise = self.parameters['bilateral_noise_amplitude_range']

        tra_coef_min = self.rng.uniform(min_noise, max_noise)
        tra_coef_max = self.rng.uniform(tra_coef_min, max_noise)

        # perform the affine transformation
        rescaled_sum = (tra_coef_max + tra_coef_min) / 2 + (tra_coef_max - tra_coef_min) / 2 * norm_sum

        # as the last step, we add random, constant offsets
        const_offsets = self.rng.normal(loc=0., scale=self.parameters['rand_noise_sigma'], size=self.Nt)
        total_noise = rescaled_sum + const_offsets[:, None, None]
        if return_noise_components:
            # reconstruct linear, quadratic and turbulent components in cm
            '''min_sum, max_sum = np.min(sum_noise), np.max(sum_noise)
            norm_sum = 2 * ((sum_noise - min_sum) / (max_sum - min_sum)) - 1

            # affine transform of the whole sum
            rescaled_sum = (tra_coef_max + tra_coef_min) / 2 + (tra_coef_max - tra_coef_min) / 2 * norm_sum

            # distribute contributions proportionally
            comp_lin_cm = rescaled_sum * (a * norm_topo_lin) / sum_noise
            comp_qua_cm = rescaled_sum * (b * norm_topo_qua) / sum_noise
            comp_turb_cm = rescaled_sum * (c * norm_turbulent) / sum_noise'''

            # Apply the same weights as in the original sum
            weighted_topo_lin = a * norm_topo_lin
            weighted_topo_qua = b * norm_topo_qua
            weighted_turbulent = c * norm_turbulent

            # Now apply the SAME normalization parameters from sum_noise
            min_sum = np.min(sum_noise)
            max_sum = np.max(sum_noise)

            # Normalize each component using the sum's min/max
            norm_topo_lin_component = 2 * ((weighted_topo_lin - min_sum) / (max_sum - min_sum)) - 1
            norm_topo_qua_component = 2 * ((weighted_topo_qua - min_sum) / (max_sum - min_sum)) - 1
            norm_turbulent_component = 2 * ((weighted_turbulent - min_sum) / (max_sum - min_sum)) - 1

            # Apply the same affine transformation coefficients
            comp_lin_cm = (tra_coef_max + tra_coef_min) / 2 + (
                    tra_coef_max - tra_coef_min) / 2 * norm_topo_lin_component
            comp_qua_cm = (tra_coef_max + tra_coef_min) / 2 + (
                    tra_coef_max - tra_coef_min) / 2 * norm_topo_qua_component
            comp_turb_cm = (tra_coef_max + tra_coef_min) / 2 + (
                    tra_coef_max - tra_coef_min) / 2 * norm_turbulent_component

            return total_noise, topo, (comp_lin_cm, comp_qua_cm, comp_turb_cm)
        return total_noise, topo

    def add_pre_existing_faults(self, batch_x, fault_list):
        p = self.rng.permutation(self.parameters['batch_size'])
        for b_idx in range(self.parameters['batch_size']):
            if self.rng.uniform() < self.parameters['secondary_fault_proba']:
                rot_angle = self.rng.choice(self.fault_rotation_angles)
                rotated_img = rotate(fault_list[p[b_idx]][0, :, :, 0], rot_angle, reshape=False)
                batch_x[b_idx, :, :, :, 0] += rotated_img[None, :, :]
        return batch_x

    def add_transient_faults(self, batch_x, fault_list, return_components=False):
        transient_faults = np.zeros_like(batch_x)
        p = self.rng.permutation(self.parameters['batch_size'])
        for b_idx in range(self.parameters['batch_size']):
            if self.rng.uniform() < self.parameters['transient_tertiary_fault_proba']:
                frame_limit = self.Nt - 1
                temp_sig_t = self.rng.integers(1, frame_limit - 1)
                max_length = frame_limit - temp_sig_t
                temp_sig_length = self.rng.integers(1, min(max_length, 3))
                rot_angle = self.rng.choice(self.fault_rotation_angles[1:])
                rotated_img = rotate(fault_list[p[b_idx]][0, :, :, 0], rot_angle, reshape=False)
                batch_x[b_idx, temp_sig_t:temp_sig_t + temp_sig_length, :, :, 0] += rotated_img
                transient_faults[b_idx, temp_sig_t:temp_sig_t + temp_sig_length, :, :, 0] += rotated_img
        if return_components:
            return batch_x, transient_faults
        return batch_x

    def add_glitchy_pixels(self, batch_x):
        for b_idx in range(self.parameters['batch_size']):
            glitch_fraction = 10 ** (self.rng.uniform(*self.parameters['glitch_frac_exp_lim']))
            for t in range(self.Nt):
                num_pixels = int(self.N * self.N * glitch_fraction)
                if num_pixels > 0:
                    row_idx = self.rng.integers(self.N, size=num_pixels)
                    col_idx = self.rng.integers(self.N, size=num_pixels)
                    rnd_values = self.rng.uniform(-1, 1, size=num_pixels) * self.parameters[
                        'glitchy_pixel_amplitude_max']
                    batch_x[b_idx, t, row_idx, col_idx, 0] = rnd_values
        return batch_x

    def add_decorrelation_areas(self, batch_x):
        buggy_fraction = self.rng.uniform(high=self.parameters['buggy_patches_frac_max'])
        for b_idx in range(self.parameters['batch_size']):
            if self.rng.uniform() < buggy_fraction:
                # we create a realistic surface to model incoherent areas etc.
                # we use the same mask for the whole time series (as in real data). For each acquisition, we have
                # a different noise distribution (but on the same mask)
                surf = self.artificial_self_affine_surface(
                    10. ** (-self.rng.integers(*self.parameters['sigma_roughness_exponent_range'])),
                    self.rng.uniform(*self.parameters['hurst_coefficient_range']))
                normsurf = (surf - surf.min()) / (surf.max() - surf.min())
                buggy_mask = normsurf > self.rng.uniform(low=self.parameters['buggy_surf_mask_min_thresh'])
                for t in range(self.Nt):
                    batch_x[b_idx, t, buggy_mask, 0] = self.rng.uniform(
                        *self.parameters['decorrelation_pixel_value_range'], size=buggy_mask.sum())
        return batch_x

    def generate_batch_data(self, return_components=False):
        """
        Create noisy samples of synthetic InSAR time-series for training
        * batch_size : number of time series to create
        * n_faults : number of different dislocation model. Each prediction will be re-used
                    (batch_size//n_faults) times, rotated and overlaid with different noise structures
        """

        batch_size, n_faults = self.parameters['batch_size'], self.parameters['n_faults']

        batch_x = np.zeros((batch_size, self.Nt, self.N, self.N, 1), dtype=self.dtype)
        batch_y = np.zeros((batch_size, self.Nt, self.N, self.N, 1), dtype=self.dtype)
        batch_cum_y = np.zeros((batch_size, 1, self.N, self.N, 1), dtype=self.dtype)
        batch_topo = np.zeros((batch_size, 1, self.N, self.N, 1), dtype=self.dtype)
        if return_components:
            batch_comp = np.zeros((batch_size, self.Nt, self.N, self.N, 5), dtype=self.dtype)

        assert batch_size % n_faults == 0, "ERROR : change parameters, batch_size should be a multiple of n_faults"

        # Compute deformation
        for i in range(n_faults):  # iterate over different disloc model
            n_same_fault = batch_size // n_faults
            fault_defo = np.zeros((self.Nt, self.N, self.N, 1), dtype=self.dtype)
            orig_fault_defo = np.zeros((self.Nt, self.N, self.N, 1), dtype=self.dtype)

            # Set environment
            self.build_los()

            # generate a new fault only fault_proba*100 % of the time
            fault_bool = self.rng.uniform() < self.parameters['fault_proba']
            if fault_bool:
                # get fault deformation
                rescaled_fault_defo, original_fault_defo = self.model_fault_deformation()
                fault_defo += rescaled_fault_defo
                orig_fault_defo += original_fault_defo
            elif not fault_bool and self.verbose:
                print("No fault for sample {} to {}".format(i * n_same_fault, (i + 1) * n_same_fault - 1))
            for j in range(n_same_fault):
                # cum_y = fault_defo[self.Nt - 1].copy()[None, ...]
                cum_y = orig_fault_defo[self.Nt - 1].copy()[None, ...]  # may be useful for curriculum learning
                x = fault_defo.copy()

                y = x.copy()  # we denoise the whole time series, not the cumulative # maybe in future add a choice
                # with given probability, add a Mogi source
                if self.rng.uniform() < self.parameters['mogi_proba']:
                    mogi_source = self.model_volcano_deformation()
                    x += mogi_source.copy()
                    if return_components:
                        batch_comp[j + i * n_same_fault, ..., 4] = mogi_source.squeeze()

                # Rotate the fault randomly to create more sample cheaply
                theta = self.rng.choice(self.fault_rotation_angles)
                x = np.array([rotate(e[:, :, 0], theta, reshape=False) for e in x], dtype=self.dtype)
                y = np.array([rotate(e[:, :, 0], theta, reshape=False) for e in y], dtype=self.dtype)
                cum_y = rotate(cum_y[0, :, :, 0], theta, reshape=False)

                if return_components:
                    total_noise, topography, components = self.model_noise_contributions(i * n_same_fault + j,
                                                                                         return_noise_components=True)
                    for cnum, component in enumerate(components):
                        '''from utils.tensorboard_plot_utils import create_fig_time_series
                        from matplotlib import pyplot as plt
                        print(batch_x.shape, component.shape, batch_y.shape)
                        create_fig_time_series(total_noise[np.newaxis, ..., np.newaxis],
                                               component[np.newaxis, ..., np.newaxis],
                                               component[np.newaxis, ..., np.newaxis], 0)
                        plt.show()'''
                        batch_comp[j + i * n_same_fault, ..., cnum] = component
                else:
                    total_noise, topography = self.model_noise_contributions(i * n_same_fault + j)

                x += total_noise

                batch_x[j + i * n_same_fault, ..., 0] = x
                batch_topo[j + i * n_same_fault, 0, ..., 0] = topography
                batch_y[j + i * n_same_fault, ..., 0] = y
                batch_cum_y[j + i * n_same_fault, 0, ..., 0] = cum_y

        p = self.rng.permutation(batch_size)
        batch_x = batch_x[p]  # DATA
        batch_y = batch_y[p]  # TRUE
        batch_topo = batch_topo[p]
        batch_cum_y = batch_cum_y[p]
        if return_components:
            batch_comp = batch_comp[p]

        # Add a pre-existing fault (permute and rotate to diversify samples) -> from other samples
        batch_x = self.add_pre_existing_faults(batch_x, batch_cum_y)

        # Add temporary fault-like signal
        if return_components:
            batch_x, temp_faults = self.add_transient_faults(batch_x, batch_cum_y, return_components=True)
            batch_comp[..., 3] = temp_faults.squeeze()
        else:
            batch_x = self.add_transient_faults(batch_x, batch_cum_y)

        # Add glitchy pixels
        batch_x = self.add_glitchy_pixels(batch_x)

        # Add buggy patches
        batch_x = self.add_decorrelation_areas(batch_x)
        if return_components:
            return batch_x, batch_y, batch_topo, batch_comp
        return batch_x, batch_y, batch_topo
