import h5py

from kito import DATASETS, H5Dataset


@DATASETS.register('H5InSARDataset')
class H5InSARDataset(H5Dataset):

    def __init__(self, path: str):
        super().__init__(path)

        self.dataset_topo = None

    def _lazy_load(self):
        if self.dataset_data is None or self.dataset_topo is None or self.dataset_labels is None:
            try:
                self.h5file = h5py.File(self.file_path, 'r')
                self.dataset_data = self.h5file["data"]
                self.dataset_topo = self.h5file["topo"]
                self.dataset_labels = self.h5file["labels"]
            except (OSError, KeyError) as e:
                raise RuntimeError(f"Failed to load H5 file '{self.file_path}': {e}")

    def _load_sample(self, index):
        """Load sample from HDF5 file with lazy loading."""
        self._lazy_load()
        return (self.dataset_data[index], self.dataset_topo[index]), self.dataset_labels[index]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove HDF5 file handle and dataset references
        state['dataset_data'] = None
        state['dataset_labels'] = None
        state['dataset_topo'] = None
        if 'h5file' in state:
            del state['h5file']
        return state
