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

    '''def forward(self, pred, target):
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
        return total_loss'''

    def forward(self, pred, target):
        mask_target, regress_target = target
        mask_output, regress_output = pred

        total_loss = torch.tensor(0.0, device=regress_target.device, dtype=regress_target.dtype)

        if self.lambda_bce > 0.:
            mask_loss = self.bce_loss(mask_output, mask_target)
            weighted_bce = self.lambda_bce * mask_loss
            total_loss = total_loss + weighted_bce
            print(f"BCE: {mask_loss.item():.6f}, weighted: {weighted_bce.item():.6f}")

        if self.lambda_l2 > 0.:
            reconstruction_loss = self.mse_loss(regress_output, regress_target)
            weighted_l2 = self.lambda_l2 * reconstruction_loss
            total_loss = total_loss + weighted_l2
            print(f"L2: {reconstruction_loss.item():.6f}, weighted: {weighted_l2.item():.6f}")

        if self.lambda_ssim > 0.:
            loss_ssim = self.ssim_loss(regress_output, regress_target)
            weighted_ssim = self.lambda_ssim * loss_ssim
            total_loss = total_loss + weighted_ssim
            print(f"SSIM: {loss_ssim.item():.6f}, weighted: {weighted_ssim.item():.6f}")

        print(f"Total: {total_loss.item():.6f}\n")
        return total_loss

