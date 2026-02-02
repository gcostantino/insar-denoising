import multiprocessing
import os
import time
from multiprocessing import Process, Manager
from queue import Empty
from random import shuffle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from config import datagen_params
from datagenerator.DataGeneratorV2 import DataGeneratorV2
from datagenerator.PureNoiseGeneratorV2 import PureNoiseGeneratorV2
from utils.pretext_utils import mask_timeseries_with_random_square
from utils.tensorboard_plot_utils_old import create_fig_time_series


def test():
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    frame = DataGenerator(params, dtype=np.float32, verbose=False)
    x, y, topo = frame.generate_batch_data(plot=False)

    print(x.shape, y.shape, topo.shape)

    for b_idx in range(frame.parameters['batch_size']):
        create_fig_time_series(x, np.zeros_like(x), y, b_idx)
        plt.show()


def generate_batch(batch_id, buffer, params, batch_size):
    generator = DataGenerator(params, verbose=False, precomputed_arrays=True)
    batch_x, batch_y, batch_topo = generator.generate_batch_data(plot=False)
    buffer.put((batch_id, batch_x, batch_y, batch_topo))
    # print(f'batch ID {batch_id} put into buffer')


def generate_and_save_batch(batch_id, basepath, params, v2, all_sources):
    if v2:
        generator = DataGeneratorV2(params, dtype=float, verbose=False)
    else:
        generator = DataGenerator(params, dtype=float, verbose=False)
    if all_sources:
        batch_x, batch_y, batch_topo, sources = generator.generate_batch_data(return_components=True)
    else:
        batch_x, batch_y, batch_topo = generator.generate_batch_data()

    while np.isnan(batch_x).sum() > 0:
        if all_sources:
            batch_x, batch_y, batch_topo, sources = generator.generate_batch_data(return_components=True)
        else:
            batch_x, batch_y, batch_topo = generator.generate_batch_data()
    # normalize topography
    min_topo, max_topo = np.min(batch_topo, axis=(2, 3), keepdims=True), np.max(batch_topo, axis=(2, 3), keepdims=True)
    norm_batch_topo = (batch_topo - min_topo) / (max_topo - min_topo)
    if all_sources:
        np.savez(os.path.join(basepath, f'{batch_id}'), batch_x=batch_x.astype(np.float32),
                 batch_y=batch_y.astype(np.float32), batch_topo=norm_batch_topo.astype(np.float32),
                 noise_sources=sources.astype(np.float32))
    else:
        np.savez(os.path.join(basepath, f'{batch_id}'), batch_x=batch_x.astype(np.float32),
                 batch_y=batch_y.astype(np.float32), batch_topo=norm_batch_topo.astype(np.float32))


def generate_and_save_batch_real_noise(batch_id, basepath, params):
    generator = DataGeneratorSurrogateNoise(params, dtype=float, verbose=False)
    batch_x, batch_y, batch_topo = generator.generate_batch_data()
    while np.isnan(batch_x).sum() > 0:
        batch_x, batch_y, batch_topo = generator.generate_batch_data()
    # normalize topography
    min_topo, max_topo = np.min(batch_topo, axis=(2, 3), keepdims=True), np.max(batch_topo, axis=(2, 3), keepdims=True)
    norm_batch_topo = (batch_topo - min_topo) / (max_topo - min_topo)
    np.savez(os.path.join(basepath, f'{batch_id}'), batch_x=batch_x.astype(np.float32),
             batch_y=batch_y.astype(np.float32), batch_topo=norm_batch_topo.astype(np.float32))


def generate_and_save_noise_batch(batch_id, basepath, params):
    generator = PureNoiseGenerator(params, dtype=float, verbose=False)
    batch_x, batch_y, batch_topo = generator.generate_batch_data()
    # normalize topography
    min_topo, max_topo = np.min(batch_topo, axis=(2, 3), keepdims=True), np.max(batch_topo, axis=(2, 3), keepdims=True)
    norm_batch_topo = (batch_topo - min_topo) / (max_topo - min_topo)
    masked_x = np.copy(batch_x)
    for i in range(len(batch_x)):
        if np.random.uniform() < params['masking_proba']:
            masked_img, _ = mask_timeseries_with_random_square(batch_x[i].squeeze(), min_size=params['min_square_size'],
                                                               max_size=params['max_square_size'])
            masked_x[i, ..., 0] = masked_img

    np.savez(os.path.join(basepath, f'{batch_id}'), batch_x=masked_x.astype(np.float32),
             batch_y=batch_y.astype(np.float32), batch_topo=norm_batch_topo.astype(np.float32))


def generate_and_save_noise_batch_v2(batch_id, basepath, params, all_sources):
    generator = PureNoiseGeneratorV2(params, dtype=float, verbose=False)

    if all_sources:
        batch_x, batch_y, batch_topo, sources = generator.generate_batch_data(return_components=True)
    else:
        batch_x, batch_y, batch_topo = generator.generate_batch_data()

    # batch_x, batch_y, batch_topo = generator.generate_batch_data()
    # normalize topography
    min_topo, max_topo = np.min(batch_topo, axis=(2, 3), keepdims=True), np.max(batch_topo, axis=(2, 3), keepdims=True)
    norm_batch_topo = (batch_topo - min_topo) / (max_topo - min_topo)

    if params['subtract_first_frame']:
        batch_x = batch_x - batch_x[:, 0:1, ...]
        batch_y = batch_y - batch_y[:, 0:1, ...]
    for i in range(len(batch_x)):
        if np.random.uniform() < params['masking_proba']:
            masked_img, _ = mask_timeseries_with_random_square(batch_x[i].squeeze(), min_size=params['min_square_size'],
                                                               max_size=params['max_square_size'])
            batch_x[i, ..., 0] = masked_img
    if not all_sources:
        np.savez(os.path.join(basepath, f'{batch_id}'), batch_x=batch_x.astype(np.float32),
                 batch_y=batch_y.astype(np.float32), batch_topo=norm_batch_topo.astype(np.float32))
    else:
        np.savez(os.path.join(basepath, f'{batch_id}'), batch_x=batch_x.astype(np.float32),
                 batch_y=batch_y.astype(np.float32), batch_topo=norm_batch_topo.astype(np.float32),
                 noise_sources=sources.astype(np.float32))


def write_batch(buffer, hdf5_path):
    print('Writer activated!')
    with h5py.File(hdf5_path, "a") as f:
        while True:
            # print('Estimated buffer size:', buffer.qsize())
            item = buffer.get()
            if item is None:
                break
            batch_id, batch_x, batch_y, batch_topo = item
            batch_size = len(batch_x)
            f["data"][batch_id * batch_size: (batch_id + 1) * batch_size] = batch_x
            f["labels"][batch_id * batch_size: (batch_id + 1) * batch_size] = batch_y
            f["topo"][batch_id * batch_size: (batch_id + 1) * batch_size] = batch_topo
            # print(f"Batch {batch_id} saved to HDF5!")


def write_batch_timeout(buffer, hdf5_path, flush_interval=20.0):
    print('Writer activated!')
    with h5py.File(hdf5_path, "a") as f:
        batch_items = []
        last_flush_time = time.time()
        while True:
            try:
                # Try to get an item from the queue with a timeout.
                item = buffer.get(timeout=flush_interval)
                if item is None:
                    # Sentinel received – break out.
                    break
                batch_items.append(item)
            except Empty:
                # No new items for flush_interval seconds.
                pass
            # If it's been a while since the last flush, and we have items, flush them.
            if batch_items and (time.time() - last_flush_time) >= flush_interval:
                _write_batches(batch_items, f)
                batch_items = []  # Reset the list.
                last_flush_time = time.time()
        # Flush any remaining items.
        if batch_items:
            _write_batches(batch_items, f)
    print("Writer finished writing all batches to disk!")


def _write_batches(batch_items, f):
    # Write each batch in the accumulated list.
    for batch_id, batch_x, batch_y, batch_topo in batch_items:
        batch_size = len(batch_x)
        start = batch_id * batch_size
        end = start + batch_size
        f["data"][start:end] = batch_x
        f["labels"][start:end] = batch_y
        f["topo"][start:end] = batch_topo
        # Optionally, print a message for debugging:
        # print(f"Batch {batch_id} written to disk.")


def generate_and_save_hdf():
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes
    # Define the HDF5 file
    HDF5_FILE_PATH = "/data/giuseppe/synthetic_database_insarnet2.h5"
    # Define dataset parameters
    total_samples = 2 ** 26  # 67M
    batch_size = params['batch_size']
    num_batches = total_samples // batch_size
    num_cpus = multiprocessing.cpu_count()
    # n_already_completed = 32572 + 1124 + 1124 + 1124
    n_already_completed = 0
    if n_already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1), dtype="float32",
                             chunks=(batch_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(batch_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(batch_size, 1, params['N'], params['N'], 1))
    m = Manager()
    q = m.Queue(maxsize=1000)  # buffer size at least 10 times higher than # writers
    p = Process(target=write_batch_timeout, args=(q, HDF5_FILE_PATH,))
    p.start()
    Parallel(n_jobs=-2, verbose=True)(
        delayed(generate_batch)(i, q, params, batch_size) for i in range(n_already_completed, num_batches))
    q.put(None)
    p.join()


def generate_and_save_separate_batches(params, total_samples, basepath, n_already_completed, v2=False,
                                       all_sources=False, n_jobs=-2):
    batch_size = params['batch_size']
    num_batches = total_samples // batch_size

    Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(generate_and_save_batch)(i, basepath, params, v2, all_sources) for i in
        range(n_already_completed, num_batches))


def generate_and_save_separate_batches_real_noise(params, total_samples, basepath, n_already_completed):
    batch_size = params['batch_size']
    num_batches = total_samples // batch_size

    Parallel(n_jobs=-2, verbose=True)(
        delayed(generate_and_save_batch_real_noise)(i, basepath, params) for i in
        range(n_already_completed, num_batches))


def generate_and_save_separate_noise_batches(params, total_samples, basepath, n_already_completed):
    batch_size = params['batch_size']
    num_batches = total_samples // batch_size

    Parallel(n_jobs=-2, verbose=True)(
        delayed(generate_and_save_noise_batch)(i, basepath, params) for i in range(n_already_completed, num_batches))


def generate_and_save_separate_noise_batches_v2(params, total_samples, basepath, n_already_completed,
                                                v2=True, all_sources=False, n_jobs=-2):
    batch_size = params['batch_size']
    num_batches = total_samples // batch_size

    Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(generate_and_save_noise_batch_v2)(i, basepath, params, all_sources) for i in
        range(n_already_completed, num_batches))


'''def precomputed_params():
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}
    tmp_generator = DataGenerator(params, verbose=False)
    xs, ys = tmp_generator.xs, tmp_generator.ys
    Xs, Ys, Zs = tmp_generator.Xs, tmp_generator.Ys, tmp_generator.Zs
    x_noise, y_noise, distances = tmp_generator.x_noise, tmp_generator.y_noise, tmp_generator.distances
    params['xs'] = xs
    params['ys'] = ys
    params['Xs'] = Xs
    params['Ys'] = Ys
    params['Zs'] = Zs
    params['x_noise'] = x_noise
    params['y_noise'] = y_noise
    params['distances'] = distances
    return params'''

'''def add_batches_to_hdf(batch_idx_start, batch_idx_end, hdf5_path, dset_batch_size, batch_size=1024):
    # current_index = 0
    current_index = batch_idx_start * batch_size
    with h5py.File(hdf5_path, "a") as f:
        for batch_id in range(batch_idx_start, batch_idx_end):
            print(f'Batch ID #{batch_id}/{batch_idx_end}')
            with np.load(f'/data/giuseppe/synthetic_database_insarnet_batches/{batch_id}.npz') as data:
                batch_x, batch_y, batch_topo = data['batch_x'], data['batch_y'], data['batch_topo']
            # batch_size = len(batch_x)
            assert batch_size % dset_batch_size == 0
            num_subchunks = batch_size // dset_batch_size
            for i in range(num_subchunks):
                start_sub = i * dset_batch_size
                end_sub = start_sub + dset_batch_size
                f["data"][current_index: current_index + dset_batch_size] = batch_x[start_sub: end_sub]
                f["labels"][current_index: current_index + dset_batch_size] = batch_y[start_sub: end_sub]
                f["topo"][current_index: current_index + dset_batch_size] = batch_topo[start_sub: end_sub]
                current_index += dset_batch_size
                # print(f"Wrote samples {current_index - dset_batch_size} to {current_index}")
            # delete batch after writing it into the dataset
            os.remove(f'/data/giuseppe/synthetic_database_insarnet_batches/{batch_id}.npz')
        # print(f"Batch {batch_id} saved to HDF5!")'''


def add_batches_to_hdf(batch_idx_start, batch_idx_end, hdf5_path, dset_batch_size, batch_size=1024, all_sources=False):
    # Compute starting index in the final HDF5 dataset.
    current_index = batch_idx_start * batch_size

    # Gather and shuffle batch IDs
    batch_ids = list(range(batch_idx_start, batch_idx_end))
    shuffle(batch_ids)

    with h5py.File(hdf5_path, "a") as f:
        for idd, batch_id in enumerate(batch_ids):
            if idd % max(1, int(len(batch_ids) * 0.1)) == 0:
                print(f'Processing batch #{idd}/{batch_idx_end}')
            batch_filepath = os.path.join('/data/giuseppe/synthetic_database_insarnet_batches', f'{batch_id}.npz')
            with np.load(batch_filepath) as data:
                batch_x, batch_y, batch_topo = data['batch_x'], data['batch_y'], data['batch_topo']
                if all_sources:
                    batch_sources = data['noise_sources']

            # Shuffle samples within this batch.
            indices = np.arange(batch_x.shape[0])
            np.random.shuffle(indices)
            batch_x = batch_x[indices]
            batch_y = batch_y[indices]
            batch_topo = batch_topo[indices]

            # Check that batch_size is divisible by dset_batch_size.
            assert batch_size % dset_batch_size == 0
            num_subchunks = batch_size // dset_batch_size

            for i in range(num_subchunks):
                start_sub = i * dset_batch_size
                end_sub = start_sub + dset_batch_size
                f["data"][current_index: current_index + dset_batch_size] = batch_x[start_sub: end_sub]
                f["labels"][current_index: current_index + dset_batch_size] = batch_y[start_sub: end_sub]
                f["topo"][current_index: current_index + dset_batch_size] = batch_topo[start_sub: end_sub]
                if all_sources:
                    f["noise_sources"][current_index: current_index + dset_batch_size] = batch_sources[
                                                                                         start_sub: end_sub]
                current_index += dset_batch_size

            # Remove the file after writing.
            os.remove(batch_filepath)


def delete_old_batches(batch_idx_start, batch_idx_end):
    for batch_id in range(batch_idx_start, batch_idx_end):
        print(f'Removing batch #{batch_id}')
        os.remove(f'/data/giuseppe/synthetic_database_insarnet_batches/{batch_id}.npz')


def gen_pretext_dataset(total_samples, already_completed=0):
    '''High-SNR dataset. No secondary nor tertiary faults. 50% noise and 50% signal'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['masking_proba'] = .5
    params['min_square_size'] = 8
    params['max_square_size'] = 64

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_pretext.h5')
    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_noise_batches(params, total_samples, tmp_folder_path,
                                             already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_pretext_dataset_v2(total_samples, already_completed=0, subtract_first_frame=False):
    '''High-SNR dataset. No secondary nor tertiary faults. 50% noise and 50% signal'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['masking_proba'] = .5
    params['min_square_size'] = 8
    params['max_square_size'] = 64
    params['subtract_first_frame'] = subtract_first_frame

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    if subtract_first_frame:
        HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_pretext_v2.1_sub1st.h5')
    else:
        HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_pretext_v2.1.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_pretext_v2.h5')
    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_noise_batches_v2(params, total_samples, tmp_folder_path,
                                                already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage1_dataset(total_samples, already_completed=0):
    '''High-SNR dataset. No secondary nor tertiary faults nor mogi sources. 50% noise and 50% signal'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.5
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.
    # params['fault_slip_range'] = (500., 1000.)  # cm
    params['surface_displacement_range'] = (100, 200)  # SNR 5 to 10 (x20, 20 being the max noise amplitude)

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage1.h5')
    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage1_dataset_v2(total_samples, already_completed=0):
    '''High-SNR dataset. No secondary nor tertiary faults nor mogi sources. 50% noise and 50% signal'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.5
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.
    # params['fault_slip_range'] = (500., 1000.)  # cm
    params['surface_displacement_range'] = (100, 200)  # SNR 5 to 10 (x20, 20 being the max noise amplitude)

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage1_v2.1.h5')
    #HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage1_v2.1.h5')
    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage2_dataset(total_samples, already_completed=0):
    '''High-SNR dataset. Secondary faults are added. No tertiary faults. No Mogi. 50% noise and 50% signal. 50% of the
    faults are secondary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.5
    params['secondary_fault_proba'] = 0.5
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.
    # params['fault_slip_range'] = (500., 1000.)  # cm
    params['surface_displacement_range'] = (100, 200)  # SNR 5 to 10 (x20, 20 being the max noise amplitude)

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage2.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage2_dataset_v2(total_samples, already_completed=0):
    '''High-SNR dataset. Secondary, tertiary faults are added as well as Mogi sources (10% each). 275% signal.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.1
    params['transient_tertiary_fault_proba'] = 0.1
    params['mogi_proba'] = 0.1
    # params['fault_slip_range'] = (500., 1000.)  # cm
    params['surface_displacement_range'] = (100, 200)  # SNR 5 to 10 (x20, 20 being the max noise amplitude)

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage2_v2.1.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage2_v2.1.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage3_dataset(total_samples, already_completed=0):
    '''High-SNR dataset. Tertiary faults and Mogi sources are added. 50% noise and 50% signal. 50% of the
    faults are secondary/tertiary/mogi.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.5
    params['secondary_fault_proba'] = 0.5
    params['transient_tertiary_fault_proba'] = 0.5
    params['mogi_proba'] = 0.5
    # params['fault_slip_range'] = (500., 1000.)  # cm
    params['surface_displacement_range'] = (100, 200)  # SNR 5 to 10 (x20, 20 being the max noise amplitude)
    # params['mogi_source_amplitude_lim'] = (100, 200)

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage3.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage4_dataset(total_samples, already_completed=0):
    '''Moderate-SNR dataset. Tertiary and secondary faults are present. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.5
    params['transient_tertiary_fault_proba'] = 0.5
    params['mogi_proba'] = 0.5
    # params['fault_slip_range'] = (100., 500.)  # cm
    params['surface_displacement_range'] = (20, 100)  # SNR 1 to 5 (x20, 20 being the max noise amplitude)
    params['mogi_source_amplitude_lim'] = (20, 100)

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage4.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage3_dataset_v2(total_samples, already_completed=0, n_jobs=-1):
    '''Moderate-SNR dataset. Tertiary and secondary faults are present. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.1
    params['transient_tertiary_fault_proba'] = 0.1
    params['mogi_proba'] = 0.1
    # params['fault_slip_range'] = (100., 500.)  # cm
    params['surface_displacement_range'] = (20, 100)  # SNR 1 to 5 (x20, 20 being the max noise amplitude)
    # params['mogi_source_amplitude_lim'] = (20, 100)

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage3_v2.1.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage3_v2.1.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True, n_jobs=n_jobs)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage5_dataset(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.5
    params['transient_tertiary_fault_proba'] = 0.5
    params['mogi_proba'] = 0.5
    params['fault_slip_range'] = (2., 500.)  # cm

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage5.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage4_dataset_v2(total_samples, already_completed=0, n_jobs=-1):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.1
    params['transient_tertiary_fault_proba'] = 0.1
    params['mogi_proba'] = 0.1
    params['fault_slip_range'] = (.1, 50.)  # cm  # (2., 500.)  # cm
    params['minimum_los_surface_displacement'] = 0.1  # centimeters
    params['normalize_temporal_slip_evolution'] = True

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage4_v2.1.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage4_v2.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True, n_jobs=n_jobs)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage4_dataset_v2_shallow_small_slip(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.
    params['depth_range'] = (0., 2.)
    params['fault_slip_range'] = (.01, 1.)  # cm

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe',
                                  'synthetic_database_insarnet_stage4_v2_depth0_2_slip_01_1cm.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage4_v2_depth0_2_slip_01_1cm.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage4_dataset_real_noise(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.1
    params['transient_tertiary_fault_proba'] = 0.1
    params['mogi_proba'] = 0.1
    params['fault_slip_range'] = (2., 500.)  # cm

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage4_real_noise.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches_real_noise(params, total_samples, tmp_folder_path,
                                                  already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage4_dataset_real_noise_strike_slip(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.
    params['fault_slip_range'] = (2., 500.)  # cm
    params['fault_rake_range'] = (170, 190)  # degrees
    params['fault_dip_range'] = (70, 90)  # degrees
    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe',
                                  'synthetic_database_insarnet_stage4_real_noise_strike_slip.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches_real_noise(params, total_samples, tmp_folder_path,
                                                  already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage5_2_dataset(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.5
    params['transient_tertiary_fault_proba'] = 0.5
    params['mogi_proba'] = 0.5
    params['fault_slip_range'] = (2., 500.)  # cm

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage5_2.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_stage5_2_shallow_dataset(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.5
    params['transient_tertiary_fault_proba'] = 0.5
    params['mogi_proba'] = 0.5
    params['fault_slip_range'] = (2., 500.)  # cm
    params['depth_range'] = (0, 30)  # km

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage5_2.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


def gen_all_sources_dataset_v2(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.1
    params['transient_tertiary_fault_proba'] = 0.1
    params['mogi_proba'] = 0.1
    params['fault_slip_range'] = (.1, 50.)  # cm  # (2., 500.)  # cm
    params['minimum_los_surface_displacement'] = 0.1  # centimeters
    params['normalize_temporal_slip_evolution'] = True

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_all_sources_v2.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage4_v2.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))
            f.create_dataset("noise_sources", shape=(total_samples, params['Nt'], params['N'], params['N'], 5),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 5))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True, all_sources=True, n_jobs=40)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size, all_sources=True)


def gen_all_sources_dataset_v2_stra_turb_old(total_samples, already_completed=0):
    '''Low-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.
    params['fault_slip_range'] = (.1, 50.)  # cm  # (2., 500.)  # cm
    params['minimum_los_surface_displacement'] = 0.1  # centimeters
    params['normalize_temporal_slip_evolution'] = True

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_all_sources_v2_stra_turb.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage4_v2.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))
            f.create_dataset("noise_sources", shape=(total_samples, params['Nt'], params['N'], params['N'], 5),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 5))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True, all_sources=True, n_jobs=40)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size, all_sources=True)


def gen_all_sources_dataset_v2_stra_turb(total_samples, already_completed=0, subtract_first_frame=False):
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['masking_proba'] = 0.
    params['subtract_first_frame'] = False

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer

    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_all_sources_v2_stra_turb.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_pretext_v2.h5')
    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))
            f.create_dataset("noise_sources", shape=(total_samples, params['Nt'], params['N'], params['N'], 5),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 5))

    generate_and_save_separate_noise_batches_v2(params, total_samples, tmp_folder_path,
                                                already_completed // datagen_batch_size, v2=True, all_sources=True,
                                                n_jobs=40)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size, all_sources=True)


def gen_test_dataset_v2(total_samples, already_completed=0, n_jobs=-1):
    '''mixed-SNR dataset. Tertiary and secondary faults are present + Mogi sources. 25% noise and 75% signal. 50% of the
    faults are secondary/tertiary.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    # tmp_folder_path = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.1
    params['transient_tertiary_fault_proba'] = 0.1
    params['mogi_proba'] = 0.1
    params['fault_slip_range'] = (.1, 100.)  # cm  # (2., 500.)  # cm
    params['minimum_los_surface_displacement'] = 0.1  # centimeters
    params['normalize_temporal_slip_evolution'] = True

    # total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_test_v2.1.h5')
    # HDF5_FILE_PATH = os.path.join(os.path.expandvars('$WORK'), 'synthetic_database_insarnet_stage4_v2.h5')

    if already_completed == 0:
        with h5py.File(HDF5_FILE_PATH, "w") as f:
            f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1),
                             dtype="float32", chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
            f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                             chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path,
                                       already_completed // datagen_batch_size, v2=True, n_jobs=n_jobs)
    add_batches_to_hdf(already_completed // datagen_batch_size, total_samples // datagen_batch_size, HDF5_FILE_PATH,
                       chunk_size)


"""def gen_stage2_dataset_v2(already_completed=0):
    '''Moderate-SNR dataset. No tertiary and secondary faults. Mogi present. 25% noise and 75% signal.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.5
    # params['fault_slip_range'] = (100., 500.)  # cm
    params['surface_displacement_range'] = (20, 100)  # SNR 1 to 5 (x20, 20 being the max noise amplitude)
    params['mogi_source_amplitude_lim'] = (20, 100)

    total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage2_v2.h5')

    with h5py.File(HDF5_FILE_PATH, "w") as f:
        f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1), dtype="float32",
                         chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
        f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1), dtype="float32",
                         chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
        f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                         chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path, 0)
    add_batches_to_hdf(0, total_samples // datagen_batch_size, HDF5_FILE_PATH, chunk_size)


def gen_stage3_dataset_v2(already_completed=0):
    '''Low-SNR dataset. No tertiary/secondary faults are present. Yes Mogi sources. 25% noise and 75% signal.'''
    tmp_folder_path = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_batches')
    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    params['fault_proba'] = 0.75
    params['secondary_fault_proba'] = 0.
    params['transient_tertiary_fault_proba'] = 0.
    params['mogi_proba'] = 0.5
    params['fault_slip_range'] = (2., 500.)  # cm

    total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']
    chunk_size = 1  # for efficient disk-cpu transfer
    HDF5_FILE_PATH = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_stage3_v2.h5')

    with h5py.File(HDF5_FILE_PATH, "w") as f:
        f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1), dtype="float32",
                         chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
        f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1), dtype="float32",
                         chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
        f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
                         chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, tmp_folder_path, 0)
    add_batches_to_hdf(0, total_samples // datagen_batch_size, HDF5_FILE_PATH, chunk_size)
"""

if __name__ == '__main__':
    # test()

    # generate_and_save_hdf()
    '''HDF5_FILE_PATH = '/data/giuseppe/synthetic_database_insarnet.h5'
    base_savepath = '/data/giuseppe/synthetic_database_insarnet_batches'

    params = {k: v for k, v in datagen_params.__dict__.items() if not k.startswith("__")}  # exclude built-in attributes

    total_samples = 2 ** 20  # 2 ** 26
    datagen_batch_size = params['batch_size']

    chunk_size = 1
    n_already_completed = 0
    # 
    # with h5py.File(HDF5_FILE_PATH, "w") as f:
    #    f.create_dataset("data", shape=(total_samples, params['Nt'], params['N'], params['N'], 1), dtype="float32",
    #                     chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
    #    f.create_dataset("labels", shape=(total_samples, params['Nt'], params['N'], params['N'], 1), dtype="float32",
    #                     chunks=(chunk_size, params['Nt'], params['N'], params['N'], 1))
    #    f.create_dataset("topo", shape=(total_samples, 1, params['N'], params['N'], 1), dtype="float32",
    #                     chunks=(chunk_size, 1, params['N'], params['N'], 1))

    generate_and_save_separate_batches(params, total_samples, base_savepath, n_already_completed)
    add_batches_to_hdf(0, total_samples // datagen_batch_size, HDF5_FILE_PATH, chunk_size)'''

    '''n_samples = 2 ** 24

    print('generating pretext dataset')
    gen_pretext_dataset(n_samples, already_completed=2 ** 20)
    print('generating stage1 dataset')
    gen_stage1_dataset(n_samples, already_completed=0)
    print('generating stage2 dataset')
    gen_stage2_dataset(n_samples, already_completed=0)
    print('generating stage3 dataset')
    gen_stage3_dataset(n_samples, already_completed=0)
    print('generating stage4 dataset')
    gen_stage4_dataset(n_samples, already_completed=0)
    print('generating stage5 dataset')
    gen_stage5_dataset(n_samples, already_completed=0)
    print('generating stage5_2 dataset')
    gen_stage5_2_dataset(n_samples, already_completed=0)'''

    # V2 data set

    n_samples_curr = 2 ** 19  # generate 500K samples for intermediate data sets
    n_samples_last_dset = 2 ** 20  # generate 1M samples for last data set
    # n_samples_last_dset = 2 ** 22  # generate 4M samples for last data set

    print('generating pretext dataset')
    #gen_pretext_dataset_v2(n_samples_curr, already_completed=0, subtract_first_frame=False)
    print('generating pretext dataset')
    #gen_pretext_dataset_v2(n_samples_curr, already_completed=0, subtract_first_frame=True)
    print('generating stage1 dataset')
    #gen_stage1_dataset_v2(n_samples_curr, already_completed=0)
    print('generating stage2 dataset')
    #gen_stage2_dataset_v2(n_samples_curr, already_completed=0)
    print('generating stage3 dataset')
    gen_stage3_dataset_v2(n_samples_curr, already_completed=0, n_jobs=60)
    print('generating stage4 dataset')
    gen_stage4_dataset_v2(n_samples_last_dset, already_completed=0, n_jobs=60)

    '''# dset from real noise
    n_samples = 2 ** 19  # generate 500K samples
    # gen_stage4_dataset_real_noise(n_samples, already_completed=0)
    gen_stage4_dataset_real_noise_strike_slip(n_samples, already_completed=0)'''

    # gen_stage4_dataset_v2_shallow_small_slip(2 ** 17, already_completed=0) # 100K samples

    # gen_pretext_dataset_v2(n_samples_curr, already_completed=0, subtract_first_frame=True)  # with no rectangles
    # gen_stage4_dataset_v2(n_samples_last_dset, already_completed=0)

    gen_test_dataset_v2(100 * 2 ** 10, already_completed=0, n_jobs=60)
    # gen_all_sources_dataset_v2(n_samples_last_dset, already_completed=0)
    # gen_all_sources_dataset_v2_stra_turb(n_samples_curr, already_completed=0)
