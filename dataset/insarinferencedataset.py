from kito import DATASETS
from kito.data.inference import InferenceDataset, _to_float32


@DATASETS.register('insar-inference-dataset')
class InSARInferenceDataset(InferenceDataset):

    def __init__(self, x, topo, y=None):
        super().__init__(x, y)
        self.topo = topo

    def _load_sample(self, index):
        x, y = super()._load_sample(index)
        topo = _to_float32(self.topo[index])
        return (x, topo), y
