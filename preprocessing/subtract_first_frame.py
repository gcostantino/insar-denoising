from kito import PREPROCESSING, Preprocessing


@PREPROCESSING.register('subtract_first_frame')
class SubtractFirstFrame(Preprocessing):

    def __init__(self):
        pass

    def __call__(self, data, labels):
        x, topo = data
        x = x.copy() - x.copy()[0:1, ...]
        labels = labels.copy() - labels.copy()[0:1, ...]
        return (x, topo), labels
