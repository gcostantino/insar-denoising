import numpy as np

from kito import PREPROCESSING, Preprocessing
from utils.pretext_utils import mask_timeseries_with_random_square


@PREPROCESSING.register('fill_the_gaps_pretext')
class FillTheGapsPretext(Preprocessing):

    def __init__(self, masking_proba, min_square_size, max_square_size):
        self.masking_proba = masking_proba
        self.min_square_size = min_square_size
        self.max_square_size = max_square_size

    def __call__(self, data, labels):
        x, topo = data

        if np.random.uniform() < self.masking_proba:
            masked_x, _ = mask_timeseries_with_random_square(x.squeeze(),
                                                             min_size=self.min_square_size,
                                                             max_size=self.max_square_size)
            return (masked_x[..., None], topo), x
        else:
            return (x, topo), x
