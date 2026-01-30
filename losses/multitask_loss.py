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

    def forward(self, pred, target):
        mask_target, regress_target = target
        mask_output, regress_output = pred

        total_loss = torch.zeros((), device=regress_target.device, dtype=regress_target.dtype)

        if self.lambda_bce > 0.:
            mask_loss = get_loss('BCEWithLogitsLoss')(mask_output, mask_target)
            total_loss += self.lambda_bce * mask_loss
        if self.lambda_l2 > 0.:
            reconstruction_loss = get_loss('L2')(regress_output, regress_target)
            total_loss += self.lambda_l2 * reconstruction_loss
        if self.lambda_ssim > 0.:
            loss_ssim = get_loss('ssim')(regress_output, regress_target)
            total_loss += self.lambda_ssim * loss_ssim
        return total_loss
