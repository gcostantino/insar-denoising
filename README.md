Source code for the paper "Denoising of InSAR time series through spatiotemporal attentive convolutional U-Net". Please cite it when using this code:


> Costantino G. & Jolivet R.: Denoising of InSAR time series through spatiotemporal attentive convolutional U-Net, in review.


The code is based on the **pytorch-kito** package ([repo](https://github.com/gcostantino/kito), [docs](https://kito.readthedocs.io/en/latest/?badge=latest)), so be sure it is installed before using this source code:

```bash
pip install pytorch-kito
```

---
## Basic usage

> Kito implements a mechanism of auto-discovery of loss functions and data preprocessing strategies, by scanning
> the `./losses/` and `./preprocessing/` packages. Do not forget to check if they are in the project root!

Everything you need is in `config/denoiser_params.py`. This python scripts creates the configuration of InSARDenoiser,
which follows the following structure:

```python
@dataclass
class InSARDenoiserConfig(KitoModuleConfig):
    """
    Base configuration container for InSARDenoiser.

    Contains all configuration sections:
    - training: Training parameters
    - model: Model architecture and settings
    - workdir: Output directories
    - data: Dataset and preprocessing
    - callbacks: Callback handling
    """
```

### Relevant parameters of the Training configuration

```python
# this is useful only if you want to re-train InSARDenoiser
learning_rate: float
n_train_epochs: int
batch_size: int
train_mode: bool  # True for training, False for inference [legacy parameter, might disappear in future releases]

distributed_training: bool = False  # 
device_id: int = 0  # Only used if training is not distributed

# Weight initialization (for training, e.g., pretrained weights)
initialize_model_with_saved_weights: bool = False

# device type initialization
device_type: str = "cuda"  # "cuda", "mps", or "cpu"
```

### Relevant parameters of the Model configuration

```python
loss: Union[str, dict] # this can be either a built-in or a dictionary (custom losses, follows pytorch-kito structure)
save_inference_to_disk: bool = False
inference_filename: str = ""
subtract_first_frame: bool = False
```

### Relevant parameters of the Workdir configuration

```python
work_directory: str = ""  # everything is happening here (e.g., logging, checkpointing)
```

### Relevant parameters of the Data configuration

```python
dataset_type: str = 'h5dataset'  # 'h5dataset', 'memdataset', or custom
dataset_path: str = ''
preprocessing: List[PreprocessingStepConfig]  # e.g., subtract first frame
# DataLoader settings
num_workers: int = 0
prefetch_factor: int = 2
pin_memory: bool = False
persistent_workers: bool = False
```

### Relevant parameters of the Callbacks configuration

```python
# === CSV Logger ===
enable_csv_logger: bool = True

# === Text Logger ===
enable_text_logger: bool = True

# === Model Checkpoint ===
enable_model_checkpoint: bool = True
enable_tensorboard: bool = False
# etc.
```

All the parameters above are handled by the software logic.
The final user just has to modify the variables in the `config/denoiser_params.py` script to customize the software behaviour.

### InSARDenoiser training example (`train.py`)

```python
denoiser = InSARDenoiser(denoiser_configuration)  # our model

engine = Engine(denoiser, denoiser_configuration)  # handles all the logic

dataset = H5InSARDataset(denoiser_configuration.data.dataset_path)  # implements the pytorch-kito logic for datasets

pipeline = GenericDataPipeline(  # the DataPipeline handles how the data is managed prior to training
    config=denoiser_configuration,
    dataset=dataset,
    preprocessing=denoiser_configuration.data.preprocessing,  # here is all stuff happening before the main training loop
)

pipeline.setup()  # finalizing the pipeline before the training
# et voilà, training can begin!
engine.fit(data_pipeline=pipeline)
```
If data parallelism is needed, the user can toggle the `distributed_training` variable (the previous code remains unchanged).
For a more efficient code, I recommend the use of `torchrun`:
```
torchrun --nproc_per_node={number_of_available_GPUs} train.py
```

### InSARDenoiser inference example
The previous code is unchanged, just replace `engine.fit()` with:
```python
# ...
engine.predict(data_pipeline=pipeline)
```
### Applying InSARDenoiser on real data (cf. `ismetpasa_denoising.py`)
```python
# suppose x, y and topo are available
# x: InSAR time series. The shape of x must be (time, height, width)
# y: ground truth, (usually not available for real datasets, so set y=None). You can set y=x also (y is never used anyways)
# topo: topography (must have the same height/width dimension as x)
# x, y, topo must fit in memory to use InSARMemDataset, otherwise create a H5InSARDataset

topo = (topo - np.nanmin(topo)) / (np.nanmax(topo) - np.nanmin(topo))  # do not forget to normalize the topography!

data, topo = data[None, ..., None], topo[None, None, ..., None]  # here, we add empty dimensions
# at the end, x will have shape (1, N_t, N, N, 1) and topo (1, 1, N, N, 1),
# where N is the image dimension (here, 128), and N_t is the temporal dimension (here, 9 frames)

dataset = InSARMemDataset(x=x, y=None, topo=topo)
pipeline = GenericDataPipeline(
    config=denoiser_configuration,
    dataset=dataset,
    preprocessing=denoiser_configuration.data.preprocessing,
)

pipeline.setup()
# for example, we want to keep the result in memory
# be sure to have:
# denoiser_configuration.model.save_inference_to_disk = False

denoiser = InSARDenoiser(denoiser_configuration)
engine = Engine(denoiser, denoiser_configuration)

# the following two lines are optional, Engine can deal with missing build and weight loading
engine.module.build()
engine.load_weights(denoiser_configuration.model.weight_load_path)

prediction = engine.predict(data_pipeline=pipeline)
```