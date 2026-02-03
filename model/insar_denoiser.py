import torch

from config.insardenoiser_config import InSARDenoiserConfig
from kito import KitoModule, Engine
from kito.config.moduleconfig import KitoModuleConfig, TrainingConfig, DataConfig, ModelConfig, WorkDirConfig, \
    CallbacksConfig
from model.unet_transformer_model import UNetTransformerModel


class InSARDenoiser(KitoModule):
    def __init__(self, config: InSARDenoiserConfig = None):
        super().__init__(config)

        self.max_encoder_features = config.model.max_encoder_features
        self.n_attention_heads = config.model.n_attention_heads
        self.transformer_dropout_rate = config.model.transformer_dropout_rate
        self.enc_conv_dropout_rate = config.model.enc_conv_dropout_rate
        self.dec_conv_dropout_rate = config.model.dec_conv_dropout_rate
        self.weight_decay = config.model.weight_decay
        self.input_data_size = config.model.input_data_size
        self.topography_size = config.model.topography_size
        self.img_size = config.model.img_size
        self.num_temporal_positions = config.model.num_temporal_positions
        self.multi_task = config.model.multi_task
        self.multi_loss = config.model.multi_loss
        self.regress_all_noise_sources = config.model.regress_all_noise_sources
        self.supervise_stratified_turbulent = config.model.supervise_stratified_turbulent

        self.standard_data_shape = (self.num_temporal_positions, self.img_size, self.img_size, 1)  # must be implemented
        # self.model_name = 'InSARDenoiser'
        self.set_model_input_size()

    def get_sample_input(self):
        sample_input = [torch.randn(1, *shape).to(self.device) for shape in self.model_input_size]
        return sample_input

    def send_data_to_device(self, data: list[torch.Tensor]):
        tensors = []
        for tensor in data:
            tensors.append(tensor.to(device=self.device))
        return tensors

    def pass_data_through_model(self, data):
        return self.model(*data)

    def freeze_encoder(self):  # should be elsewhere...
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        # self.logger.log_info('Encoder parameters frozen successfully.')

    def compute_loss(self, data_pair, y_pred, **kwargs):
        if self.multi_task or self.regress_all_noise_sources:
            target_tensors = self.send_data_to_device(data_pair[1])
            return self.apply_loss(target_tensors, y_pred, **kwargs)
        else:
            return super().compute_loss(data_pair, y_pred, **kwargs)

    def handle_model_outputs(self, outputs):
        if self.multi_task:
            return outputs[1]  # discard mask
        return super().handle_model_outputs(outputs)

    def optimizer_params(self):
        model = self.model
        return model.parameters()  # {'params': model.parameters(), 'weight_decay': self.weight_decay}  # default

    def bind_optimizer(self, *args, **kwargs):
        params_for_optimizer = self.optimizer_params()
        self.optimizer = torch.optim.Adam(params=params_for_optimizer, lr=self.learning_rate)

    def set_model_input_size(self, *args, **kwargs):
        self.model_input_size = [self.input_data_size, self.topography_size]

    def build_inner_model(self, *args, **kwargs):
        self.model = UNetTransformerModel(self.num_temporal_positions, self.enc_conv_dropout_rate,
                                          self.dec_conv_dropout_rate, self.transformer_dropout_rate,
                                          max_encoder_features=self.max_encoder_features,
                                          n_attention_heads=self.n_attention_heads,
                                          multi_task=self.multi_task,
                                          regress_all_noise_sources=self.regress_all_noise_sources,
                                          supervise_stratified_turbulent=self.supervise_stratified_turbulent)

    def _check_data_shape(self, batch):
        x, labels = batch
        data, topo = x
        if self.multi_task:
            mask, labels = labels
            if (exp_lab_shape := (
            self.batch_size, self.num_temporal_positions, self.img_size, self.img_size, 1)) != mask.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match the expected shape: ({exp_lab_shape}).")
        if self.regress_all_noise_sources:
            noise_sources, labels = labels
            if (exp_lab_shape := (
            self.batch_size, self.num_temporal_positions, self.img_size, self.img_size, 5)) != noise_sources.shape:
                raise ValueError(
                    f"Noise sources shape {noise_sources.shape} does not match the expected shape: ({exp_lab_shape}).")
        if (exp_data_shape := (
        self.batch_size, self.num_temporal_positions, self.img_size, self.img_size, 1)) != data.shape:
            raise ValueError(f"Data shape {data.shape}does not match the expected shape: ({exp_data_shape}).")
        if (exp_topo_shape := (self.batch_size, 1, self.img_size, self.img_size, 1)) != topo.shape:
            raise ValueError(f"Topography shape {topo.shape} does not match the expected shape: ({exp_topo_shape}).")
        if (exp_lab_shape := (
        self.batch_size, self.num_temporal_positions, self.img_size, self.img_size, 1)) != labels.shape:
            raise ValueError(f"Labels shape {labels.shape} does not match the expected shape: ({exp_lab_shape}).")


'''class InSARDenoiser2(BaseModule):
    """Uses custom loss from registry + runtime customization."""

    def build_inner_model(self):
        self.model = MyUNet()
        self.model_input_size = (1, 256, 256)

    def apply_loss(self, pred, target, **kwargs):
        """
        Override to pass extra inputs to loss function.

        This is still useful when loss needs runtime data
        like coherence maps, masks, epoch number, etc.
        """
        # Get coherence from kwargs (passed from training loop)
        coherence = kwargs.get('coherence', None)
        epoch = kwargs.get('epoch', 0)

        # Pass to loss function (if it supports it)
        if isinstance(self.loss, InSARPhaseLoss):
            return self.loss(pred, target, coherence=coherence)
        else:
            return self.loss(pred, target)'''

if __name__ == '__main__':
    from config.denoiser_params import denoiser_configuration
    from kito.losses import get_loss, LossRegistry
    from losses import multitask_loss

    print("Checking registration:")
    print(f"  'multi_task_loss' registered: {LossRegistry.is_registered('multi_task_loss')}")
    print(f"  'ssim' registered: {LossRegistry.is_registered('ssim')}")
    print(f"  'bce_with_logits' registered: {LossRegistry.is_registered('bce_with_logits')}")
    print(f"  'mse' registered: {LossRegistry.is_registered('mse')}")

    # Create loss
    loss = LossRegistry.create('multi_task_loss',
                               lambda_ssim=0.1,
                               lambda_bce=1.0,
                               lambda_l2=1.0)
    print(f"\n✅ Created loss: {loss}")

    # Test forward pass

    exit(0)
    denoiser = InSARDenoiser(denoiser_configuration)

    engine = Engine(denoiser, denoiser_configuration)
    exit(0)
    # engine.fit()
