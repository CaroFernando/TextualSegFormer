import torch
import torchvision
import pytorch_lightning as pl
import torchmetrics as tm

from params import *
from transformers import CLIPProcessor, CLIPModel
from ConditionedSegFormerPE import ConditionedSegFormer
from LossFunc import *

class CLIPConditionedSegFormer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

        self.segformer = ConditionedSegFormer(
            ModelParams.INCHANNELS,
            ModelParams.WIDTHS,
            ModelParams.DEPTHS,
            512,
            768,
            ModelParams.PATCH_SIZES,
            ModelParams.OVERLAP_SIZES,
            ModelParams.NUM_HEADS,
            ModelParams.EXPANSION_FACTORS,
            ModelParams.DECODER_CHANNELS,
            ModelParams.SCALE_FACTORS
        )

        self.plot_every = TrainParams.PLOT_EVERY
        self.neloss = NELoss(LossParams.ALPHA, LossParams.BETA)
        # self.neloss = FocalLoss()
        self.acc = tm.Accuracy(task="binary", threshold=LossParams.THRESHOLD)
        self.dice = DiceLoss()
        self.iou = IoULoss(LossParams.THRESHOLD)
        self.f1score = tm.F1Score(task="binary", threshold=LossParams.THRESHOLD)

        # freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, x, x_c, condition):
        condition = self.clip.text_model(condition).last_hidden_state
        pe = self.clip.vision_model(x_c).last_hidden_state

        out = self.segformer(x, pe, condition)
        return out
    
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x, x_c, condition, y = batch
        y_hat = self(x, x_c, condition)
        loss = self.neloss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        t_y_hat = torch.where(y_hat > 0.5, 1, 0).long().view(-1)
        t_y = torch.where(y > 0.5, 1, 0).long().view(-1)

        acc = self.acc(t_y_hat, t_y)
        dice = self.dice(y_hat, y)
        iou = self.iou(y_hat, y)
        f1 = self.f1score(t_y_hat, t_y)

        self.log("train_acc", acc, prog_bar=True)
        self.log("train_dice", dice, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        
        if self.global_step % self.plot_every == 0: 
            y = y.repeat(1, 3, 1, 1)
            y_hat = torch.sigmoid(y_hat)
            y_hat = y_hat.repeat(1, 3, 1, 1)
            # x_grid = torchvision.utils.make_grid(train_dataset.image_inverse_transform(x))
            # self.logger.experiment.add_image('train_sample_image', x_grid, self.global_step)
            grid = torchvision.utils.make_grid(torch.cat([y, y_hat], dim=0))
            self.logger.experiment.add_image('train_sample_mask', grid, self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x, x_c, condition, y = batch
        y_hat = self(x, x_c, condition)
        loss = self.neloss(y_hat, y)

        t_y_hat = torch.where(y_hat > 0.5, 1, 0).long().view(-1)
        t_y = torch.where(y > 0.5, 1, 0).long().view(-1)

        acc = self.acc(t_y_hat, t_y)
        dice = self.dice(y_hat, y)
        iou = self.iou(y_hat, y)
        f1 = self.f1score(t_y_hat, t_y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x, x_c, condition, y = batch
        y_hat = self(x, x_c, condition)

        t_y_hat = torch.where(y_hat > 0.5, 1, 0).long().view(-1)
        t_y = torch.where(y > 0.5, 1, 0).long().view(-1)

        acc = self.acc(t_y_hat, t_y)
        dice = self.dice(y_hat, y)
        iou = self.iou(y_hat, y)
        f1 = self.f1score(t_y_hat, t_y)

        self.log("test_acc", acc, prog_bar=True)
        self.log("test_dice", dice, prog_bar=True)
        self.log("test_iou", iou, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
    
    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed**(-0.5) * min((step+1)**(-0.5), (step+1) * warmup_steps**(-1.5))
    
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-7)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.99)
        optimizer = torch.optim.Adam(self.parameters(), lr=OptimParams.LR, betas=OptimParams.BETAS, eps=1e-7)
        warmup_steps = OptimParams.WARMUP_STEPS
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: self.calc_lr(step, 512, warmup_steps))
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }         
        }