import torch
from torch.nn import MSELoss
from torch.optim import Adam
from lightning import LightningModule
from datetime import datetime


class Base_Model(LightningModule):
    """
    Base implementation for all ML models, which assumes a sequence-to-sequence model,
    optimized with Adam and using MSE loss. It also logs the training and validation losses.
    """
    def __init__(self, in_steps=5, out_steps=5, in_channels=4, out_channels=4, lr=5e-4):
        super().__init__()

        self.in_steps = in_steps
        self.out_steps = out_steps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = lr
        self.loss_fn = MSELoss()
        self.current_val_losses = []

    def configure_optimizers(self):
        return Adam([dict(
            params=self.parameters(),
            lr=self.lr),
        ])
    
    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        train_loss = self.compute_loss(y_pred, y_true)
        self.log("Train Loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return train_loss

    # Common method for validation & test steps
    def val_test_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        loss = self.compute_loss(y_pred, y_true)
        
        return loss, y_pred, y_true

    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self.val_test_step(batch, batch_idx)
        self.log("Validation Loss", val_loss)
        self.current_val_losses.append(val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        time_str = datetime.now().strftime("%H:%M:%S")
        mean_val_loss = torch.FloatTensor(self.current_val_losses).mean()
        print(f'{time_str}: Validation Loss = {mean_val_loss}')
        self.current_val_losses.clear()

    def predict_step(self, batch, batch_idx):
        _, y_pred, y_true = self.val_test_step(batch, batch_idx)
        return y_pred, y_true