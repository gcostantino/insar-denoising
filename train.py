from config.denoiser_params import denoiser_configuration
from dataset.h5insardataset import H5InSARDataset
from kito import Engine
from kito.data import GenericDataPipeline
from model.insar_denoiser import InSARDenoiser

if __name__ == '__main__':
    denoiser = InSARDenoiser(denoiser_configuration)

    engine = Engine(denoiser, denoiser_configuration)

    dataset = H5InSARDataset(denoiser_configuration.data.dataset_path)

    pipeline = GenericDataPipeline(
        config=denoiser_configuration,
        dataset=dataset,
        preprocessing=denoiser_configuration.data.preprocessing,
    )

    pipeline.setup()

    engine.fit(data_pipeline=pipeline)
