import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


def create_trainer(training_config, callback_config, logger_config, trainer_config):
    logger = TensorBoardLogger(
        logger_config['save_dir']
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir + "/checkpoints",
        filename='best-checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=callback_config['checkpoint']['save_top_k'],
        monitor=callback_config['checkpoint']['monitor'],
        mode=callback_config['checkpoint']['mode']
    )

    early_stop_callback = EarlyStopping(
        monitor=callback_config['early_stopping']['monitor'],
        patience=callback_config['early_stopping']['patience'],
        mode=callback_config['early_stopping']['mode'],
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator=trainer_config['accelerator'],
        devices=trainer_config['devices'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        deterministic=trainer_config['deterministic'],
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 4),
        num_sanity_val_steps=0
    )
    
    return trainer


class LightningTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, loss_fn=None, scheduler_config=None, pos_weight=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(f"Using weighted BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")
        elif loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            
        self.scheduler_config = scheduler_config
        
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss
    
    def on_after_backward(self):
        # Log gradient norms
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log('grad_norm', total_norm, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        if self.scheduler_config is None:
            return optimizer
        
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.scheduler_config.get('milestones', [30, 60, 90]),
            gamma=self.scheduler_config.get('gamma', 0.5)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.scheduler_config.get('monitor', 'val_loss'),
                'interval': self.scheduler_config.get('interval', 'epoch'),
                'frequency': self.scheduler_config.get('frequency', 1)
            }
        }