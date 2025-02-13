import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import Base_Model
from einops import rearrange

# OpenSTL Imports for Temporal Attention Unit (TAU) Model
from openstl.models.simvp_model import Encoder, Decoder, MidMetaNet

class TAU_Model(Base_Model):
    def __init__(
        self, in_steps, in_channels=4, out_channels=4, lr=5e-4, hid_S=64, hid_T=256,
        N_S=4, N_T=8, spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True,
        mlp_ratio=8., drop=0.0, drop_path=0.1, alpha=0.1, tau=0.1):
        
        super().__init__(
            in_steps=in_steps,
            in_channels=in_channels, 
            out_channels=out_channels, 
            lr=lr)
        
        self.save_hyperparameters()

        self.encoder = Encoder(
            C_in=in_channels, 
            C_hid=hid_S, 
            N_S=N_S, 
            spatio_kernel=spatio_kernel_enc,
            act_inplace=act_inplace)
    
        self.hidden = MidMetaNet(
            in_steps * hid_S, hid_T, N_T, model_type="tau",
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

        self.decoder = Decoder(
            C_hid=hid_S,
            C_out=out_channels,
            N_S=N_S,
            spatio_kernel=spatio_kernel_dec,
            act_inplace=act_inplace
        )

        self.activation = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()
        self.current_val_losses = []
    

    def forward(self, x):
        # SPATIAL ENCODER
        T = self.hparams.in_steps
        x = rearrange(x, "b t c h w -> (b t) c h w")        
        enc, skip = self.encoder(x)

        # TEMPORAL ATTENTION MODULE
        z = rearrange(enc, "(b t) c h w -> b t c h w", t=T)
        hid = self.hidden(z)
        hid = rearrange(hid, "b t c h w -> (b t) c h w")

        # SPATIAL DECODER
        dec = self.decoder(hid, skip)
        dec = rearrange(dec, "(b t) c h w -> b t c h w", t=T)

        # ACTIVATION
        out = self.activation(dec)
        return out

    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T = pred_y.shape[:2]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()
    
    def compute_loss(self, y_pred, y_true):
        intra_frame_loss = self.loss_fn(y_pred, y_true)
        inter_frame_loss = self.hparams.alpha * self.diff_div_reg(
            y_pred, y_true, tau=self.hparams.tau)

        return intra_frame_loss + inter_frame_loss
