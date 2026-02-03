import torch

from kito.losses import LossRegistry, get_loss
import torch.nn as nn


@LossRegistry.register('multi_task_loss')
class MultiTaskLoss(nn.Module):

    def __init__(self, lambda_ssim, lambda_bce, lambda_l2):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_bce = lambda_bce
        self.lambda_l2 = lambda_l2

        self.bce_loss = None
        self.mse_loss = None
        self.ssim_loss = None

        if self.lambda_bce > 0.:
            self.bce_loss = LossRegistry.create('BCEWithLogitsLoss')

        if self.lambda_l2 > 0.:
            self.mse_loss = LossRegistry.create('L2')

        if self.lambda_ssim > 0.:
            self.ssim_loss = LossRegistry.create('ssim')

    def forward(self, pred, target):
        mask_target, regress_target = target
        mask_output, regress_output = pred

        total_loss = torch.zeros((), device=regress_target.device, dtype=regress_target.dtype, requires_grad=False)
        # requires_grad must be False otherwise this turns to a Leaf variable, thus no in-place operations can be done
        if self.lambda_bce > 0.:
            mask_loss = self.bce_loss(mask_output, mask_target)
            total_loss += self.lambda_bce * mask_loss
        if self.lambda_l2 > 0.:
            reconstruction_loss = self.mse_loss(regress_output, regress_target)
            total_loss += self.lambda_l2 * reconstruction_loss
        if self.lambda_ssim > 0.:
            loss_ssim = self.ssim_loss(regress_output, regress_target)
            total_loss += self.lambda_ssim * loss_ssim
        return total_loss
