"""
Author: Giuseppe Costantino, February 2025.
I decided to split the parameters into user-accessible and private ones. Please do not modify the latter ones,
as they are associated with structural properties of the methods and should not be edited by users.
"""

import os

from callbacks.insar_image_plotter import InSARTimeSeriesPlotter
from config.datagen_params import N, Nt
from config.insardenoiser_config import InSARDenoiserConfig, InSARDenoiserModelConfig
from kito import Pipeline
from kito.config.moduleconfig import TrainingConfig, PreprocessingStepConfig, WorkDirConfig, DataConfig, CallbacksConfig
from kito.losses import LossRegistry
#from losses import multitask_loss

# ************************************** work-directory-related parameters  *******************************************#
work_directory = os.path.join('/', 'data', 'giuseppe', 'insar-denoising')
# work_directory = os.path.expandvars('$WORK')

# ***************************************** training-related parameters  **********************************************#
learning_rate = 1e-4
n_train_epochs = 300
batch_size = 128
train_data_ratio = 0.8
val_data_ratio = 0.1
num_train_samples = 100_000  # 1_000_100 # 100_500  # 4_000_500  # 100_500  # use this parameter to choose a different #data wrt the training datafile content
train_verbosity_level = 2
val_verbosity_level = 2
test_verbosity_level = 2
train_mode = True  # set to False in case of inference
distributed_training = False

# ****************************************** model-related parameters  ************************************************#
transformer_dropout_rate = 0.1
enc_conv_dropout_rate = dec_conv_dropout_rate = 0.
# dec_conv_dropout_rate = 0.01
weight_decay = 0.  # 1e-5
num_temporal_positions = Nt
img_size = N
input_data_size = (num_temporal_positions, img_size, img_size, 1)
topography_size = (1, img_size, img_size, 1)
tensorboard_img_id = 'denoising'
loss = 'mean_squared_error'  # name of a loss registered in the LossRegistry
save_model_weights = True
log_to_tensorboard = True
text_logging = True
csv_logging = True
initialize_model_with_weights = True  # the model will be initialized with the weights found at {weight_load_path}
train_codename = f'train_unet_transf_bs{batch_size}_lr{learning_rate}_{weight_decay}wd_500Ksamp_128model_ResConv_normtopo_unet_{enc_conv_dropout_rate}dropout_curriculum_v2.1_stage4_sub1st_MSE0.1_BCE_multitask'  # codename for saving files and ID training
batch_idx_viz = [i for i in range(20)]  # idx of samples in a random batch to visualize in TensorBoard
save_inference_to_disk = True  # save inference to HDF5 file or store in memory

# *********************************** files- and directory-related parameters  ****************************************#
training_set_datafile = os.path.join('/', 'data', 'giuseppe', 'synthetic_database_insarnet_test_v2.1.h5')
# training_set_datafile = os.path.join(work_directory, 'synthetic_database_insarnet_stage3_v2.h5')
weight_filename = 'best_InSARDenoiser_18Dec2025-000319_train_unet_transf_bs128_lr0.0001_0.0wd_500Ksamp_128model_ResConv_normtopo_unet_0.0dropout_curriculum_v2.1_stage4_sub1st_MSE0.1_BCE_multitask.pt'  # best_InSARDenoiser_24Sep2025-144712_train_unet_transf_bs128_lr1e-07_0.0wd_1Msamp_128model_ResConv_normtopo_unet_0.0dropout_curriculum_v2_stage4_sub1st_MSE0.1_BCE_multitask.pt
weight_load_path = os.path.join(work_directory, 'weights', 'InSARDenoiser', weight_filename)
inference_filename = os.path.join('/', 'data', 'giuseppe', 'inference',
                                  'inference_curriculum_v2_1Mparams_weights_18Dec2025-000319.h5')  # './sample_pred.h5'  # be sure to include the whole path (used when save_inference_to_disk=True)

###########################################  private, structural parameters ############################################

# --------------------------------------------- GPU-specific parameters  --------------------------------------------- #
device_type = 'cuda'
chosen_gpu_id = 0  # meaning: device 'cuda:{chosen_gpu_id}'. NOT used when distributed_training = True

# ---------------------------------------- (parallel) data loading parameters  --------------------------------------- #
num_workers_dataloader = 2
prefetch_factor_dataloader = 2
use_pin_memory = True
use_persistent_workers = True
load_full_dataset_memory = False

# --------------------------------------------- model-related parameters  -------------------------------------------- #
max_encoder_features = 1 * 128
n_attention_heads = 1 * 4

# ------------------------------------------- training-related parameters  ------------------------------------------- #
subtract_first_frame = True
masking_pre_text = False
segmentation_task = False
regress_all_noise_sources = False
supervise_stratified_turbulent = False
multi_task = True
multi_loss = False
freeze_encoder = False
mask_loss_weight = 1.
structural_similarity_loss_weight = 0.  # 1.
reconstruction_loss_weight = 0.1  # .1 # 1.
# trust_loss_weight = 0.
temporal_evolution_loss_weight = 0.  # 1.

# ----------------------------------------- dataclass configuration objects  ----------------------------------------- #
preprocessing_config_list = []
if subtract_first_frame:
    preprocessing_config_list.append(PreprocessingStepConfig('subtract_first_frame', dict()))
if multi_task:
    preprocessing_config_list.append(PreprocessingStepConfig('multi_task', {'disp_fraction': 0.5}))
if masking_pre_text:
    preprocessing_config_list.append(PreprocessingStepConfig('fill_the_gaps_pretext',
                                                             {'masking_proba': .5, 'min_square_size': 8,
                                                              'max_square_size': 64}))

if multi_task:
    '''from kito.losses import get_loss
    loss = get_loss({'name':'multi_task_loss', 'lambda_ssim':structural_similarity_loss_weight,
                               'lambda_bce':mask_loss_weight, 'lambda_l2':reconstruction_loss_weight})'''
    '''loss = LossRegistry.create('multi_task_loss', lambda_ssim=structural_similarity_loss_weight,
                               lambda_bce=mask_loss_weight, lambda_l2=reconstruction_loss_weight)'''
    loss = {
        'name': 'multi_task_loss',
        'params': {
            'lambda_ssim': structural_similarity_loss_weight,
            'lambda_bce': mask_loss_weight,
            'lambda_l2': reconstruction_loss_weight
        }
    }

denoiser_configuration = InSARDenoiserConfig(
    training=TrainingConfig(
        learning_rate=learning_rate,
        n_train_epochs=n_train_epochs,
        batch_size=batch_size,
        train_mode=train_mode,

        train_verbosity_level=train_verbosity_level,
        val_verbosity_level=val_verbosity_level,
        test_verbosity_level=test_verbosity_level,
        distributed_training=distributed_training,
        device_id=chosen_gpu_id,
        initialize_model_with_saved_weights=initialize_model_with_weights,
        device_type=device_type,
    ),

    model=InSARDenoiserModelConfig(
        input_data_size=input_data_size,
        loss=loss,
        train_codename=train_codename,
        weight_load_path=weight_load_path,
        save_inference_to_disk=save_inference_to_disk,
        inference_filename=inference_filename,
        tensorboard_img_id=tensorboard_img_id,
        batch_idx_viz=batch_idx_viz,
        max_encoder_features=max_encoder_features,
        n_attention_heads=n_attention_heads,
        transformer_dropout_rate=transformer_dropout_rate,
        enc_conv_dropout_rate=enc_conv_dropout_rate,
        dec_conv_dropout_rate=dec_conv_dropout_rate,
        weight_decay=weight_decay,
        segmentation_task=segmentation_task,
        regress_all_noise_sources=regress_all_noise_sources,
        multi_task=multi_task,
        multi_loss=multi_loss,
        supervise_stratified_turbulent=supervise_stratified_turbulent,
        freeze_encoder=freeze_encoder,
        subtract_first_frame=subtract_first_frame,
        mask_loss_weight=mask_loss_weight,
        structural_similarity_loss_weight=structural_similarity_loss_weight,
        reconstruction_loss_weight=reconstruction_loss_weight,
        temporal_evolution_loss_weight=temporal_evolution_loss_weight,
        topography_size=topography_size,
        img_size=img_size,
        num_temporal_positions=num_temporal_positions,

    ),
    callbacks=CallbacksConfig(
        enable_model_checkpoint=save_model_weights,
        enable_text_logger=text_logging,
        enable_csv_logger=csv_logging,
        checkpoint_monitor='val_loss',
        checkpoint_save_best_only=True,
        checkpoint_verbose=False,
        enable_tensorboard=log_to_tensorboard,
        tensorboard_scalars=True,
        tensorboard_histograms=True,
        tensorboard_histogram_freq=1,
        tensorboard_graph=True,
        tensorboard_images=True,
        tensorboard_image_freq=1,
        tensorboard_batch_indices=batch_idx_viz,
        image_plotter_class=InSARTimeSeriesPlotter,
    ),
    data=DataConfig(
        dataset_type='H5InSARDataset',  # name of a registered dataset
        dataset_path=training_set_datafile,
        load_into_memory=load_full_dataset_memory,  # full dataset into memory
        train_ratio=train_data_ratio,
        val_ratio=val_data_ratio,
        total_samples=num_train_samples,
        preprocessing=preprocessing_config_list,
        num_workers=num_workers_dataloader,
        prefetch_factor=prefetch_factor_dataloader,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    ),
    workdir=WorkDirConfig(
        work_directory=work_directory
    )
)

del N, Nt
