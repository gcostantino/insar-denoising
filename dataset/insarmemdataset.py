from kito import MemDataset, DATASETS


@DATASETS.register('insar-memdataset')
class InSARMemDataset(MemDataset):

    def __init__(self, x, y, topo):
        super().__init__(x, y)
        self.topo = topo

    def _load_sample(self, index):
        return (self.x[index], self.topo[index]), self.y[index]
