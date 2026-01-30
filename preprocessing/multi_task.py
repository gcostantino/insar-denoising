from kito import PREPROCESSING, Preprocessing
from utils.pretext_utils import mask_los_disp_highest_disp


@PREPROCESSING.register('multi_task')
class MultiTask(Preprocessing):

    def __init__(self, disp_fraction):
        self.disp_fraction = disp_fraction

    def __call__(self, data, labels):
        labels_masked = mask_los_disp_highest_disp(labels, frac=self.disp_fraction)

        return data, (labels_masked, labels)
