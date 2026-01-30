import torch
import torch.nn as nn

from model.layers.doubleconv import DoubleConv
from model.layers.spatiotemporal_transformer import SpatioTemporalTransformer
from model.layers.temporal_embedding import TemporalEmbedding


class UNetEncoder(nn.Module):
    def __init__(self, conv_dropout_rate, n_inputs=2, max_features=128):
        super().__init__()

        n_conv_layers = 5
        n_feature_in, n_feature_out = [], []
        for i in range(n_conv_layers):
            n_feature_out.append(max_features)
            max_features = max_features // 2
        n_feature_out = n_feature_out[::-1]
        n_feature_in = [n_inputs] + n_feature_out[:-1]

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out

        self.enc1 = DoubleConv(n_feature_in[0], n_feature_out[0], conv_dropout_rate)
        self.enc2 = DoubleConv(n_feature_in[1], n_feature_out[1], conv_dropout_rate)
        self.enc3 = DoubleConv(n_feature_in[2], n_feature_out[2], conv_dropout_rate)
        self.enc4 = DoubleConv(n_feature_in[3], n_feature_out[3], conv_dropout_rate)
        self.enc5 = DoubleConv(n_feature_in[4], n_feature_out[4], conv_dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        x3 = self.pool(x2)
        x3 = self.enc3(x3)
        x4 = self.pool(x3)
        x4 = self.enc4(x4)
        x5 = self.pool(x4)
        x5 = self.enc5(x5)
        return x1, x2, x3, x4, x5


class UNetDecoder(nn.Module):
    def __init__(self, conv_dropout_rate, n_features_in, n_features_out, add_final_conv=True):
        super().__init__()
        self.add_final_conv = add_final_conv
        n_features_in, n_features_out = n_features_out[::-1], n_features_in[::-1]
        self.up1 = nn.ConvTranspose2d(n_features_in[0], n_features_out[0], kernel_size=2,
                                      stride=2)  # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(n_features_in[0], n_features_out[0], conv_dropout_rate)
        self.up2 = nn.ConvTranspose2d(n_features_in[1], n_features_out[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(n_features_in[1], n_features_out[1], conv_dropout_rate)
        self.up3 = nn.ConvTranspose2d(n_features_in[2], n_features_out[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(n_features_in[2], n_features_out[2], conv_dropout_rate)
        self.up4 = nn.ConvTranspose2d(n_features_in[3], n_features_out[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(n_features_in[3], n_features_out[3], conv_dropout_rate)
        if self.add_final_conv:
            self.final_conv = nn.Conv2d(n_features_in[4], 1, kernel_size=1)

    def forward(self, x, enc1, enc2, enc3, enc4):
        x = self.up1(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec1(x)
        x = self.up2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec2(x)
        x = self.up3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)
        x = self.up4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec4(x)
        if self.add_final_conv:
            return self.final_conv(x)
        return x


class UNetTransformerModel(nn.Module):
    def __init__(self, num_temporal_positions, enc_conv_dropout_rate, dec_conv_dropout_rate, transformer_dropout_rate,
                 max_encoder_features=128, n_attention_heads=4, multi_task=False,
                 regress_all_noise_sources=False, supervise_stratified_turbulent=False):
        super().__init__()
        self.multi_task = multi_task
        self.regress_all_noise_sources = regress_all_noise_sources
        self.supervise_stratified_turbulent = supervise_stratified_turbulent
        self.encoder = UNetEncoder(enc_conv_dropout_rate, max_features=max_encoder_features)
        self.temporal_embedding = TemporalEmbedding(num_temporal_positions, max_encoder_features)
        self.transformer = SpatioTemporalTransformer(max_encoder_features, max_encoder_features, axis='both',
                                                     causal=False, dropout=transformer_dropout_rate,
                                                     return_attention=False, n_heads=n_attention_heads)
        self.decoder = UNetDecoder(dec_conv_dropout_rate, self.encoder.n_feature_in, self.encoder.n_feature_out,
                                   add_final_conv=(not multi_task and not regress_all_noise_sources))
        if self.multi_task:
            self.mask_head = nn.Conv2d(self.encoder.n_feature_out[0], 1, kernel_size=1)  # logits for mask
            self.regression_head = nn.Conv2d(self.encoder.n_feature_out[0], 1,
                                             kernel_size=1)  # reconstructed amplitude
        if self.regress_all_noise_sources:
            self.stratified_linear_head = nn.Conv2d(1 + self.encoder.n_feature_out[0], 1, kernel_size=1)
            self.stratified_quadratic_head = nn.Conv2d(1 + self.encoder.n_feature_out[0], 1, kernel_size=1)
            self.turbulent_head = nn.Conv2d(self.encoder.n_feature_out[0], 1, kernel_size=1)
            if not self.supervise_stratified_turbulent:
                self.transient_fault_head = nn.Conv2d(self.encoder.n_feature_out[0], 1, kernel_size=1)
                self.mogi_head = nn.Conv2d(self.encoder.n_feature_out[0], 1, kernel_size=1)
                self.fault_head = nn.Conv2d(self.encoder.n_feature_out[0], 1, kernel_size=1)  # signal

    def forward(self, x, topography, **kwargs):
        """
        Forward pass of the UNetTransformerModel
        :param x: input InSAR time series. Expected shape: (batch, n_time_steps, n_pixels, n_pixels, 1)
        :param topography: topography image. Expected shape: (batch, 1, n_pixels, n_pixels, 1)
        :return: denoised input time series. Expected shape: (batch, n_time_steps, n_pixels, n_pixels, 1)
        """
        x = torch.cat([x, topography.expand(x.shape)], dim=-1)  # concatenate topography prior to the CNN
        B, Nt, Nx, Ny, C = x.shape  # B, C, Nt, Nx, Ny = x.shape
        x = x.view(B * Nt, C, Nx, Ny)  # reshape for CNN
        enc1, enc2, enc3, enc4, bottleneck = self.encoder(x)
        B_Nt, C_bott, Nx_pool, Ny_pool = bottleneck.shape
        bottleneck = bottleneck.view(B_Nt // Nt, Nt, Nx_pool * Ny_pool,
                                     C_bott)  # reshape again for Transformer. Spatial dimension: nodes from flattened pixels
        bottleneck = self.temporal_embedding(
            bottleneck)  # we embed time here, since we do "temporal then spatial" self-attention
        bottleneck = self.transformer(bottleneck)
        B, Nt, N, C_tra = bottleneck.shape
        bottleneck = bottleneck.view(B * Nt, C_tra, Nx_pool, Ny_pool)  # reshape back for CNNs in expanding path
        # return self.decoder(bottleneck[:, :, :, :, 0], enc1, enc2, enc3, enc4)
        decoded = self.decoder(bottleneck, enc1, enc2, enc3, enc4)
        '''if self.segmentation:
            decoded = torch.sigmoid(decoded)'''
        if self.multi_task:
            mask_logits = self.mask_head(decoded)
            disp_pred = self.regression_head(decoded)
            return mask_logits.view(B, Nt, Nx, Ny, 1), disp_pred.view(B, Nt, Nx, Ny, 1)
        if self.regress_all_noise_sources:
            strat_input = torch.cat([decoded, topography.squeeze(-1).expand(B, Nt, Nx, Ny).reshape(B * Nt, 1, Nx, Ny)], dim=1)
            stratified_lin_pred = self.stratified_linear_head(strat_input).view(B, Nt, Nx, Ny)
            stratified_qua_pred = self.stratified_quadratic_head(strat_input).view(B, Nt, Nx, Ny)
            turbulent_pred = self.turbulent_head(decoded).view(B, Nt, Nx, Ny)
            if not self.supervise_stratified_turbulent:
                transient_fault_pred = self.transient_fault_head(decoded).view(B, Nt, Nx, Ny)
                mogi_pred = self.mogi_head(decoded).view(B, Nt, Nx, Ny)
                fault_pred = self.fault_head(decoded).view(B, Nt, Nx, Ny, 1)
                return stratified_lin_pred, stratified_qua_pred, turbulent_pred, transient_fault_pred, mogi_pred, fault_pred
            tmp = torch.zeros_like(turbulent_pred)
            return stratified_lin_pred, stratified_qua_pred, turbulent_pred, tmp, tmp, tmp.unsqueeze(-1)  # correct that
        return decoded.view(B, Nt, Nx, Ny, 1)
