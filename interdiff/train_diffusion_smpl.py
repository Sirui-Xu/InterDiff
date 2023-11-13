import os
from datetime import datetime
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from tools import rotvec_to_rotmat
from model.diffusion_smpl import create_model_and_diffusion
from data.dataset_smpl import Dataset, OBJECT_PATH, MODEL_PATH
from data.utils import SIMPLIFIED_MESH
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
import functools
from diffusion.resample import LossAwareSampler
from diffusion.resample import create_named_schedule_sampler
from render.mesh_viz import visualize_body_obj
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

class LitInteraction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
        smpl_male = SMPL_Layer(center_idx=0, gender='male', num_betas=10,
                                model_root=str(MODEL_PATH), hands=True).to(device)
        smpl_female = SMPL_Layer(center_idx=0, gender='female', num_betas=10,
                                model_root=str(MODEL_PATH), hands=True).to(device)
        self.body_model = {'male': smpl_male, 'female': smpl_female}

        self.model, self.diffusion = create_model_and_diffusion(args)
        self.use_ddp = False
        self.ddp_model = self.model
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

    def on_train_start(self) -> None:
        #     backup trainer and model
        shutil.copy('./train_diffusion_smpl.py', str(save_dir / 'train_diffusion_smpl.py'))
        shutil.copy('./model/diffusion_smpl.py', str(save_dir / 'diffusion_smpl.py'))
        shutil.copy('./data/dataset_smpl.py', str(save_dir / 'dataset_smpl.py'))
        shutil.copy('./diffusion/gaussian_diffusion.py', str(save_dir / 'gaussian_diffusion.py'))
        return

    def l2(self, a, b):
        # assuming a.shape == b.shape == seqlen, bs, N
        loss = torch.nn.MSELoss(reduction='none')(a, b)
        loss = loss.mean(dim=[0, 2]) 
        return loss

    def forward_backward(self, batch, cond):
        t, weights = self.schedule_sampler.sample(batch.shape[0], device)

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,  # [bs, ch, image_size, image_size]
            t,  # [bs](int) sampled timesteps
            model_kwargs=cond,
        )

        pred, gt = compute_losses()
        body_pred, obj_pred = torch.split(pred.squeeze(1).permute(2, 0, 1).contiguous(), self.args.smpl_dim+3, dim=2)
        body_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), self.args.smpl_dim+3, dim=2)
        
        loss_dict = dict()
        weighted_loss_dict = dict()

        T, B, nJ = body_pred[:, :, :-3].shape
        nJ = nJ // 6
        body_rot = body_pred[:, :, :-3]
        body_rot_gt = body_gt[:, :, :-3]
        obj_rot = obj_pred[:, :, :-3]
        obj_rot_gt = obj_gt[:, :, :-3]

        loss_body_rot_past = self.l2(body_rot[:self.args.past_len], body_rot_gt[:self.args.past_len])
        loss_body_nonrot_past = self.l2(body_pred[:self.args.past_len, :, -3:], body_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_past = self.l2(obj_rot[:self.args.past_len], obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_past = self.l2(obj_pred[:self.args.past_len, :, -3:], obj_gt[:self.args.past_len, :, -3:])

        loss_body_rot_v_past = self.l2(body_rot[1:self.args.past_len+1]-body_rot[:self.args.past_len], body_rot_gt[1:self.args.past_len+1]-body_rot_gt[1:self.args.past_len+1]) +\
                               self.l2(body_rot[1:self.args.past_len]-body_rot[:self.args.past_len-1], body_rot[2:self.args.past_len+1]-body_rot[1:self.args.past_len])
        loss_body_nonrot_v_past = self.l2(body_pred[1:self.args.past_len+1, :, -3:]-body_pred[:self.args.past_len, :, -3:], body_gt[1:self.args.past_len+1, :, -3:]-body_gt[1:self.args.past_len+1, :, -3:]) +\
                                  self.l2(body_pred[1:self.args.past_len, :, -3:]-body_pred[:self.args.past_len-1, :, -3:], body_pred[2:self.args.past_len+1, :, -3:]-body_pred[1:self.args.past_len, :, -3:])

        loss_obj_rot_v_past = self.l2(obj_rot[1:self.args.past_len+1]-obj_rot[:self.args.past_len], obj_rot_gt[1:self.args.past_len+1]-obj_rot_gt[1:self.args.past_len+1]) +\
                              self.l2(obj_rot[1:self.args.past_len]-obj_rot[:self.args.past_len-1], obj_rot[2:self.args.past_len+1]-obj_rot[1:self.args.past_len])
        loss_obj_nonrot_v_past = self.l2(obj_pred[1:self.args.past_len+1, :, -3:]-obj_pred[:self.args.past_len, :, -3:], obj_gt[1:self.args.past_len+1, :, -3:]-obj_gt[1:self.args.past_len+1, :, -3:]) +\
                                 self.l2(obj_pred[1:self.args.past_len, :, -3:]-obj_pred[:self.args.past_len-1, :, -3:], obj_pred[2:self.args.past_len+1, :, -3:]-obj_pred[1:self.args.past_len, :, -3:])

        loss_body_rot_future = self.l2(body_rot[self.args.past_len:], body_rot_gt[self.args.past_len:])
        loss_body_nonrot_future = self.l2(body_pred[self.args.past_len:, :, -3:], body_gt[self.args.past_len:, :, -3:])

        loss_obj_rot_future = self.l2(obj_rot[self.args.past_len:], obj_rot_gt[self.args.past_len:])
        loss_obj_nonrot_future = self.l2(obj_pred[self.args.past_len:, :, -3:], obj_gt[self.args.past_len:, :, -3:])

        loss_body_rot_v_future = self.l2(body_rot[self.args.past_len:]-body_rot[self.args.past_len-1:-1], body_rot_gt[self.args.past_len:]-body_rot_gt[self.args.past_len:]) +\
                                 self.l2(body_rot[self.args.past_len-1:-2]-body_rot[self.args.past_len:-1], body_rot[self.args.past_len:-1]-body_rot[self.args.past_len+1:])
        loss_body_nonrot_v_future = self.l2(body_pred[self.args.past_len:, :, -3:]-body_pred[self.args.past_len-1:-1, :, -3:], body_gt[self.args.past_len:, :, -3:]-body_gt[self.args.past_len:, :, -3:]) +\
                                    self.l2(body_pred[self.args.past_len-1:-2, :, -3:]-body_pred[self.args.past_len:-1, :, -3:], body_pred[self.args.past_len:-1, :, -3:]-body_pred[self.args.past_len+1:, :, -3:])

        loss_obj_rot_v_future = self.l2(obj_rot[self.args.past_len:]-obj_rot[self.args.past_len-1:-1], obj_rot_gt[self.args.past_len:]-obj_rot_gt[self.args.past_len:]) +\
                                self.l2(obj_rot[self.args.past_len-1:-2]-obj_rot[self.args.past_len:-1], obj_rot[self.args.past_len:-1]-obj_rot[self.args.past_len+1:])
        loss_obj_nonrot_v_future = self.l2(obj_pred[self.args.past_len:, :, -3:]-obj_pred[self.args.past_len-1:-1, :, -3:], obj_gt[self.args.past_len:, :, -3:]-obj_gt[self.args.past_len:, :, -3:]) +\
                                   self.l2(obj_pred[self.args.past_len-1:-2, :, -3:]-obj_pred[self.args.past_len:-1, :, -3:], obj_pred[self.args.past_len:-1, :, -3:]-obj_pred[self.args.past_len+1:, :, -3:])

        loss_dict.update(dict(
                        body_rot_past=loss_body_rot_past,
                        body_nonrot_past=loss_body_nonrot_past,
                        obj_rot_past=loss_obj_rot_past,
                        obj_nonrot_past=loss_obj_nonrot_past,
                        body_rot_v_past=loss_body_rot_v_past,
                        body_nonrot_v_past=loss_body_nonrot_v_past,
                        obj_rot_v_past=loss_obj_rot_v_past,
                        obj_nonrot_v_past=loss_obj_nonrot_v_past,
                        body_rot_future=loss_body_rot_future,
                        body_nonrot_future=loss_body_nonrot_future,
                        obj_rot_future=loss_obj_rot_future,
                        obj_nonrot_future=loss_obj_nonrot_future,
                        body_rot_v_future=loss_body_rot_v_future,
                        body_nonrot_v_future=loss_body_nonrot_v_future,
                        obj_rot_v_future=loss_obj_rot_v_future,
                        obj_nonrot_v_future=loss_obj_nonrot_v_future,
                        ))

        weighted_loss_dict.update(dict(
                                body_rot_past=loss_body_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_past=loss_body_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                body_rot_v_past=loss_body_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_v_past=loss_body_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                body_rot_future=loss_body_rot_future * self.args.weight_smplx_rot,
                                body_nonrot_future=loss_body_nonrot_future * self.args.weight_smplx_nonrot,
                                obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                body_rot_v_future=loss_body_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body_nonrot_v_future=loss_body_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                ))

        loss = sum(list(weighted_loss_dict.values()))

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, loss.detach()
            )

        loss = (loss * weights).mean()
        self.log_loss_dict(
            self.diffusion, t, weighted_loss_dict, loss
        )
        return loss, body_pred, obj_pred, body_gt, obj_gt

    def log_loss_dict(self, diffusion, ts, losses, loss):
        self.log('train_loss', loss, prog_bar=False)
        for key, values in losses.items():
            self.log(key, values.mean().item(), prog_bar=True)
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"{key}_q{quartile}", sub_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=list(self.model.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        return ({'optimizer': optimizer,
                 })

    def calc_val_loss(self, body_pred, body_gt, obj_pred, obj_gt, batch):
        loss_dict = dict()
        weighted_loss_dict = dict()

        T, B, nJ = body_pred[:, :, :-3].shape
        nJ = nJ // 3
        body_rot = rotvec_to_rotmat(body_pred[:, :, :-3]).view(T, B, nJ * 9)
        body_rot_gt = rotvec_to_rotmat(body_gt[:, :, :-3]).view(T, B, nJ * 9)
        obj_rot = rotvec_to_rotmat(obj_pred[:, :, :-3]).view(T, B, 9)
        obj_rot_gt = rotvec_to_rotmat(obj_gt[:, :, :-3]).view(T, B, 9)

        loss_body_rot_past = torch.nn.MSELoss(reduction='mean')(body_rot[:self.args.past_len], body_rot_gt[:self.args.past_len])
        loss_body_nonrot_past = torch.nn.MSELoss(reduction='mean')(body_pred[:self.args.past_len, :, -3:], body_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:self.args.past_len], obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:self.args.past_len, :, -3:], obj_gt[:self.args.past_len, :, -3:])

        loss_body_rot_v_past = torch.nn.MSELoss(reduction='mean')(body_rot[1:self.args.past_len+1]-body_rot[:self.args.past_len], body_rot_gt[1:self.args.past_len+1]-body_rot_gt[:self.args.past_len])
        loss_body_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(body_pred[1:self.args.past_len+1, :, -3:]-body_pred[:self.args.past_len, :, -3:], body_gt[1:self.args.past_len+1, :, -3:]-body_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_v_past = torch.nn.MSELoss(reduction='mean')(obj_rot[1:self.args.past_len+1]-obj_rot[:self.args.past_len], obj_rot_gt[1:self.args.past_len+1]-obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(obj_pred[1:self.args.past_len+1, :, -3:]-obj_pred[:self.args.past_len, :, -3:], obj_gt[1:self.args.past_len+1, :, -3:]-obj_gt[:self.args.past_len, :, -3:])

        loss_body_rot_future = torch.nn.MSELoss(reduction='mean')(body_rot[self.args.past_len:], body_rot_gt[self.args.past_len:])
        loss_body_nonrot_future = torch.nn.MSELoss(reduction='mean')(body_pred[self.args.past_len:, :, -3:], body_gt[self.args.past_len:, :, -3:])

        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:], obj_rot_gt[self.args.past_len:])
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:], obj_gt[self.args.past_len:, :, -3:])

        loss_body_rot_v_future = torch.nn.MSELoss(reduction='mean')(body_rot[self.args.past_len:]-body_rot[self.args.past_len-1:-1], body_rot_gt[self.args.past_len:]-body_rot_gt[self.args.past_len-1:-1])
        loss_body_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(body_pred[self.args.past_len:, :, -3:]-body_pred[self.args.past_len-1:-1, :, -3:], body_gt[self.args.past_len:, :, -3:]-body_gt[self.args.past_len-1:-1, :, -3:])

        loss_obj_rot_v_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:]-obj_rot[self.args.past_len-1:-1], obj_rot_gt[self.args.past_len:]-obj_rot_gt[self.args.past_len-1:-1])
        loss_obj_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:]-obj_pred[self.args.past_len-1:-1, :, -3:], obj_gt[self.args.past_len:, :, -3:]-obj_gt[self.args.past_len-1:-1, :, -3:])

        loss_dict.update(dict(
                        body_rot_past=loss_body_rot_past,
                        body_nonrot_past=loss_body_nonrot_past,
                        obj_rot_past=loss_obj_rot_past,
                        obj_nonrot_past=loss_obj_nonrot_past,
                        body_rot_v_past=loss_body_rot_v_past,
                        body_nonrot_v_past=loss_body_nonrot_v_past,
                        obj_rot_v_past=loss_obj_rot_v_past,
                        obj_nonrot_v_past=loss_obj_nonrot_v_past,
                        body_rot_future=loss_body_rot_future,
                        body_nonrot_future=loss_body_nonrot_future,
                        obj_rot_future=loss_obj_rot_future,
                        obj_nonrot_future=loss_obj_nonrot_future,
                        body_rot_v_future=loss_body_rot_v_future,
                        body_nonrot_v_future=loss_body_nonrot_v_future,
                        obj_rot_v_future=loss_obj_rot_v_future,
                        obj_nonrot_v_future=loss_obj_nonrot_v_future,
                        ))

        weighted_loss_dict.update(dict(
                                body_rot_past=loss_body_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_past=loss_body_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                body_rot_v_past=loss_body_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_v_past=loss_body_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                body_rot_future=loss_body_rot_future * self.args.weight_smplx_rot,
                                body_nonrot_future=loss_body_nonrot_future * self.args.weight_smplx_nonrot,
                                obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                body_rot_v_future=loss_body_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body_nonrot_v_future=loss_body_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                ))

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def calc_loss(self, body_pred, body_gt, obj_pred, obj_gt, batch):
        loss_dict = dict()
        weighted_loss_dict = dict()

        T, B, nJ = body_gt[:, :, :-3].shape
        nJ = nJ // 3
        body_rot = rotvec_to_rotmat(body_pred[:, :, :, :-3]).view(self.args.diverse_samples, T, B, nJ * 9)
        body_rot_gt = rotvec_to_rotmat(body_gt[:, :, :-3]).view(T, B, nJ * 9).unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        obj_rot = rotvec_to_rotmat(obj_pred[:, :, :, :-3]).view(self.args.diverse_samples, T, B, 9)
        obj_rot_gt = rotvec_to_rotmat(obj_gt[:, :, :-3]).view(T, B, 9).unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        body_gt = body_gt.unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        obj_gt = obj_gt.unsqueeze(0).repeat(self.args.diverse_samples, 1, 1, 1)
        
        loss_body_rot_past = torch.nn.MSELoss(reduction='mean')(body_rot[:, :self.args.past_len], body_rot_gt[:, :self.args.past_len])
        loss_body_nonrot_past = torch.nn.MSELoss(reduction='mean')(body_pred[:, :self.args.past_len, :, -3:], body_gt[:, :self.args.past_len, :, -3:])

        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:, :self.args.past_len], obj_rot_gt[:, :self.args.past_len])
        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, :self.args.past_len, :, -3:])

        loss_body_rot_v_past = torch.nn.MSELoss(reduction='mean')(body_rot[:, 1:self.args.past_len+1]-body_rot[:, :self.args.past_len], body_rot_gt[:, 1:self.args.past_len+1]-body_rot_gt[:, :self.args.past_len])
        loss_body_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(body_pred[:, 1:self.args.past_len+1, :, -3:]-body_pred[:, :self.args.past_len, :, -3:], body_gt[:, 1:self.args.past_len+1, :, -3:]-body_gt[:, :self.args.past_len, :, -3:])

        loss_obj_rot_v_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:, 1:self.args.past_len+1]-obj_rot[:, :self.args.past_len], obj_rot_gt[:, 1:self.args.past_len+1]-obj_rot_gt[:, :self.args.past_len])
        loss_obj_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:, 1:self.args.past_len+1, :, -3:]-obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, 1:self.args.past_len+1, :, -3:]-obj_gt[:, :self.args.past_len, :, -3:])

        loss_body_rot_future = torch.nn.MSELoss(reduction='mean')(body_rot[:, self.args.past_len:], body_rot_gt[:, self.args.past_len:])
        loss_body_nonrot_future = torch.nn.MSELoss(reduction='mean')(body_pred[:, self.args.past_len:, :, -3:], body_gt[:, self.args.past_len:, :, -3:])

        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(obj_rot[:, self.args.past_len:], obj_rot_gt[:, self.args.past_len:])
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(obj_pred[:, self.args.past_len:, :, -3:], obj_gt[:, self.args.past_len:, :, -3:])

        loss_body_rot_v_future = torch.nn.MSELoss(reduction='mean')(body_rot[:, self.args.past_len+1:]-body_rot[:, self.args.past_len:-1], body_rot_gt[:, self.args.past_len+1:]-body_rot_gt[:, self.args.past_len:-1])
        loss_body_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(body_pred[:, self.args.past_len+1:, :, -3:]-body_pred[:, self.args.past_len:-1, :, -3:], body_gt[:, self.args.past_len+1:, :, -3:]-body_gt[:, self.args.past_len:-1, :, -3:])

        loss_obj_rot_v_future = torch.nn.MSELoss(reduction='mean')(obj_rot[:, self.args.past_len+1:]-obj_rot[:, self.args.past_len:-1], obj_rot_gt[:, self.args.past_len+1:]-obj_rot_gt[:, self.args.past_len:-1])
        loss_obj_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(obj_pred[:, self.args.past_len+1:, :, -3:]-obj_pred[:, self.args.past_len:-1, :, -3:], obj_gt[:, self.args.past_len+1:, :, -3:]-obj_gt[:, self.args.past_len:-1, :, -3:])

        loss_body_rot_past_min = torch.nn.MSELoss(reduction='none')(body_rot[:, :self.args.past_len], body_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body_nonrot_past_min = torch.nn.MSELoss(reduction='none')(body_pred[:, :self.args.past_len, :, -3:], body_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_past_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, :self.args.past_len], obj_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_past_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_body_rot_v_past_min = torch.nn.MSELoss(reduction='none')(body_rot[:, 1:self.args.past_len+1]-body_rot[:, :self.args.past_len], body_rot_gt[:, 1:self.args.past_len+1]-body_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body_nonrot_v_past_min = torch.nn.MSELoss(reduction='none')(body_pred[:, 1:self.args.past_len+1, :, -3:]-body_pred[:, :self.args.past_len, :, -3:], body_gt[:, 1:self.args.past_len+1, :, -3:]-body_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_v_past_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, 1:self.args.past_len+1]-obj_rot[:, :self.args.past_len], obj_rot_gt[:, 1:self.args.past_len+1]-obj_rot_gt[:, :self.args.past_len]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_v_past_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, 1:self.args.past_len+1, :, -3:]-obj_pred[:, :self.args.past_len, :, -3:], obj_gt[:, 1:self.args.past_len+1, :, -3:]-obj_gt[:, :self.args.past_len, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_body_rot_future_min = torch.nn.MSELoss(reduction='none')(body_rot[:, self.args.past_len:], body_rot_gt[:, self.args.past_len:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body_nonrot_future_min = torch.nn.MSELoss(reduction='none')(body_pred[:, self.args.past_len:, :, -3:], body_gt[:, self.args.past_len:, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_future_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, self.args.past_len:], obj_rot_gt[:, self.args.past_len:]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_future_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, self.args.past_len:, :, -3:], obj_gt[:, self.args.past_len:, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_body_rot_v_future_min = torch.nn.MSELoss(reduction='none')(body_rot[:, self.args.past_len+1:]-body_rot[:, self.args.past_len:-1], body_rot_gt[:, self.args.past_len+1:]-body_rot_gt[:, self.args.past_len:-1]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_body_nonrot_v_future_min = torch.nn.MSELoss(reduction='none')(body_pred[:, self.args.past_len+1:, :, -3:]-body_pred[:, self.args.past_len:-1, :, -3:], body_gt[:, self.args.past_len+1:, :, -3:]-body_gt[:, self.args.past_len:-1, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_obj_rot_v_future_min = torch.nn.MSELoss(reduction='none')(obj_rot[:, self.args.past_len+1:]-obj_rot[:, self.args.past_len:-1], obj_rot_gt[:, self.args.past_len+1:]-obj_rot_gt[:, self.args.past_len:-1]).mean(dim=[1, 3]).min(dim=0)[0].mean()
        loss_obj_nonrot_v_future_min = torch.nn.MSELoss(reduction='none')(obj_pred[:, self.args.past_len+1:, :, -3:]-obj_pred[:, self.args.past_len:-1, :, -3:], obj_gt[:, self.args.past_len+1:, :, -3:]-obj_gt[:, self.args.past_len:-1, :, -3:]).mean(dim=[1, 3]).min(dim=0)[0].mean()

        loss_dict.update(dict(
                        body_rot_past=loss_body_rot_past,
                        body_nonrot_past=loss_body_nonrot_past,
                        obj_rot_past=loss_obj_rot_past,
                        obj_nonrot_past=loss_obj_nonrot_past,
                        body_rot_v_past=loss_body_rot_v_past,
                        body_nonrot_v_past=loss_body_nonrot_v_past,
                        obj_rot_v_past=loss_obj_rot_v_past,
                        obj_nonrot_v_past=loss_obj_nonrot_v_past,
                        body_rot_future=loss_body_rot_future,
                        body_nonrot_future=loss_body_nonrot_future,
                        obj_rot_future=loss_obj_rot_future,
                        obj_nonrot_future=loss_obj_nonrot_future,
                        body_rot_v_future=loss_body_rot_v_future,
                        body_nonrot_v_future=loss_body_nonrot_v_future,
                        obj_rot_v_future=loss_obj_rot_v_future,
                        obj_nonrot_v_future=loss_obj_nonrot_v_future,
                        body_rot_past_min=loss_body_rot_past_min,
                        body_nonrot_past_min=loss_body_nonrot_past_min,
                        obj_rot_past_min=loss_obj_rot_past_min,
                        obj_nonrot_past_min=loss_obj_nonrot_past_min,
                        body_rot_v_past_min=loss_body_rot_v_past_min,
                        body_nonrot_v_past_min=loss_body_nonrot_v_past_min,
                        obj_rot_v_past_min=loss_obj_rot_v_past_min,
                        obj_nonrot_v_past_min=loss_obj_nonrot_v_past_min,
                        body_rot_future_min=loss_body_rot_future_min,
                        body_nonrot_future_min=loss_body_nonrot_future_min,
                        obj_rot_future_min=loss_obj_rot_future_min,
                        obj_nonrot_future_min=loss_obj_nonrot_future_min,
                        body_rot_v_future_min=loss_body_rot_v_future_min,
                        body_nonrot_v_future_min=loss_body_nonrot_v_future_min,
                        obj_rot_v_future_min=loss_obj_rot_v_future_min,
                        obj_nonrot_v_future_min=loss_obj_nonrot_v_future_min,
                        ))

        weighted_loss_dict.update(dict(
                                body_rot_past=loss_body_rot_past * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_past=loss_body_nonrot_past * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                body_rot_v_past=loss_body_rot_v_past * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_v_past=loss_body_nonrot_v_past * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                body_rot_future=loss_body_rot_future * self.args.weight_smplx_rot,
                                body_nonrot_future=loss_body_nonrot_future * self.args.weight_smplx_nonrot,
                                obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                body_rot_v_future=loss_body_rot_v_future * self.args.weight_v * self.args.weight_smplx_rot,
                                body_nonrot_v_future=loss_body_nonrot_v_future * self.args.weight_v * self.args.weight_smplx_nonrot,
                                obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                ))

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def _common_step(self, batch, batch_idx, mode):
        embedding, gt = self.model._get_embeddings(batch)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}}
        if mode == 'train':
            loss, body_pred, obj_pred, body_gt, obj_gt = self.forward_backward(gt, model_kwargs)
            return loss
        elif mode == 'valid':
            model_kwargs['y']['inpainted_motion'] = gt
            model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                    device=device)  # True means use gt motion
            model_kwargs['y']['inpainting_mask'][:, :, :, self.args.past_len:] = False  # do inpainting in those frames
            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(self.model, gt.shape, clip_denoised=False, model_kwargs=model_kwargs)
            body_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), self.args.smpl_dim+3, dim=2)
            body_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), self.args.smpl_dim+3, dim=2)
            T, B, _ = body_pred[:, :, :-3].shape
            body_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            body_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            idx_pad = list(range(self.args.past_len)) + [self.args.past_len - 1] * self.args.future_len
            hand_pose = torch.cat([frame['smplfit_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float()
            body_pred = torch.cat([body_rot, hand_pose[idx_pad], body_pred[:, :, -3:]], dim=2)
            body_gt = torch.cat([body_rot_gt, hand_pose, body_gt[:, :, -3:]], dim=2)
            obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)
            obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)
            loss, loss_dict, weighted_loss_dict = self.calc_val_loss(body_pred, body_gt, obj_pred, obj_gt, batch=batch)

            render_interval = 100
            if (batch_idx % render_interval == 0) and (((self.current_epoch % self.args.render_epoch) == self.args.render_epoch - 1) or self.args.debug):
                self.visualize(body_pred, obj_pred, body_gt, obj_gt, batch, batch_idx, mode, 0)
            return loss, loss_dict, weighted_loss_dict

        elif mode == 'test':
            model_kwargs['y']['inpainted_motion'] = gt
            model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                    device=device)  # True means use gt motion
            model_kwargs['y']['inpainting_mask'][:, :, :, self.args.past_len:] = False  # do inpainting in those frames
            sample_fn = self.diffusion.p_sample_loop
            body_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), self.args.smpl_dim+3, dim=2)
            idx_pad = list(range(self.args.past_len)) + [self.args.past_len - 1] * self.args.future_len
            hand_pose = torch.cat([frame['smplfit_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float()
            # body_gt = torch.cat([body_gt[:, :, :-3], hand_pose, body_gt[:, :, -3:]], dim=2)
            body_preds, obj_preds = [], []
            T, B, _ = body_gt[:, :, :-3].shape
            body_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
            body_gt = torch.cat([body_rot_gt, hand_pose, body_gt[:, :, -3:]], dim=2)
            obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)
            for idx in range(self.args.diverse_samples):
                sample = sample_fn(self.model, gt.shape, clip_denoised=False, model_kwargs=model_kwargs)
                body_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), self.args.smpl_dim+3, dim=2)
                body_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
                obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
                body_pred = torch.cat([body_rot, hand_pose[idx_pad], body_pred[:, :, -3:]], dim=2)
                obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)
                body_preds.append(body_pred.unsqueeze(0))
                obj_preds.append(obj_pred.unsqueeze(0))
            body_preds = torch.cat(body_preds, dim=0)
            obj_preds = torch.cat(obj_preds, dim=0)
            loss, loss_dict, weighted_loss_dict = self.calc_loss(body_preds, body_gt, obj_preds, obj_gt, batch=batch)

            render_interval = 100
            if (batch_idx % render_interval == 0):
                for idx in range(self.args.diverse_samples):
                    if idx == 0:
                        self.visualize(body_preds[idx], obj_preds[idx], body_gt, obj_gt, batch, batch_idx, mode, idx)
                    else:
                        self.visualize(body_preds[idx], obj_preds[idx], None, None, batch, batch_idx, mode, idx)
            return loss, loss_dict, weighted_loss_dict

    def visualize(self, body_pred, obj_pred, body_gt, obj_gt, batch, batch_idx, mode, idx):
        with torch.no_grad():
            body = body_pred.detach().clone()
            obj = obj_pred.detach().clone()

            # visualize
            export_file = Path.joinpath(save_dir, 'render')
            export_file.mkdir(exist_ok=True, parents=True)
            rend_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_{}.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate, idx))
            
            smpl = self.body_model[batch['gender'][0]]
            verts, jtr, _, _ = smpl(body[:, 0, :-3], 
                                    th_betas=torch.cat([record['smplfit_params']['betas'][0:1] for record in batch['frames']], dim=0), 
                                    th_trans=body[:, 0, -3:])
            jtr = jtr.detach().cpu().numpy()

            verts = verts.detach().cpu().numpy()
            
            faces = smpl.th_faces.cpu().numpy()
            obj_verts = []

            if body_gt is not None and obj_gt is not None:
                body_gt = body_gt.detach().clone()
                obj_gt = obj_gt.detach().clone()
                rend_gt_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_gt.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))
                verts_gt, jtr_gt, _, _ = smpl(body_gt[:, 0, :-3], 
                                        th_betas=torch.cat([record['smplfit_params']['betas'][0:1] for record in batch['frames']], dim=0), 
                                        th_trans=body_gt[:, 0, -3:])
                jtr_gt = jtr_gt.detach().cpu().numpy()
                verts_gt = verts_gt.detach().cpu().numpy()
                obj_verts_gt = []

            for t, record in (enumerate(batch['frames'])):
                mesh_obj = Mesh()
                mesh_obj.load_from_file(os.path.join(OBJECT_PATH, SIMPLIFIED_MESH[batch['obj_name'][0]]))
                mesh_obj_v = mesh_obj.v.copy()
                # center the meshes
                center = np.mean(mesh_obj.v, 0)
                mesh_obj.v = mesh_obj.v - center
                angle, trans = obj[t][0][:-3].detach().cpu().numpy(), obj[t][0][-3:].detach().cpu().numpy()
                rot = Rotation.from_rotvec(angle).as_matrix()
                # transform canonical mesh to fitting
                mesh_obj.v = np.matmul(mesh_obj.v, rot.T) + trans
                obj_verts.append(mesh_obj.v)

                if body_gt is not None and obj_gt is not None:
                    # center the meshes
                    center = np.mean(mesh_obj_v, 0)
                    mesh_obj_v = mesh_obj_v - center
                    angle, trans = obj_gt[t][0][:-3].detach().cpu().numpy(), obj_gt[t][0][-3:].detach().cpu().numpy()
                    rot = Rotation.from_rotvec(angle).as_matrix()
                    # transform canonical mesh to fitting
                    mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
                    obj_verts_gt.append(mesh_obj_v)
    
            m1 = visualize_body_obj(verts, faces, np.array(obj_verts), mesh_obj.f, past_len=self.args.past_len, save_path=rend_video_path, sample_rate=self.args.sample_rate)
            if body_gt is not None and obj_gt is not None:
                m2 = visualize_body_obj(verts_gt, faces, np.array(obj_verts_gt), mesh_obj.f, past_len=self.args.past_len, save_path=rend_gt_video_path, sample_rate=self.args.sample_rate)

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'test')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--model", type=str, default='Diffusion')
    parser.add_argument("--use_pointnet2", type=int, default=1)
    parser.add_argument("--num_obj_keypoints", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=1)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument('--star_graph', default=False, action='store_true')

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_template", type=float, default=0.1)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_contact", type=float, default=0)
    parser.add_argument("--weight_dist", type=float, default=1)
    parser.add_argument("--weight_penetration", type=float, default=0)  #10

    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.2)
    parser.add_argument("--weight_obj_rot", type=float, default=0.1)
    parser.add_argument("--weight_obj_nonrot", type=float, default=0.2)
    parser.add_argument("--weight_past", type=float, default=1)
    parser.add_argument("--weight_jtr", type=float, default=0.1)
    parser.add_argument("--weight_jtr_v", type=float, default=500)
    parser.add_argument("--weight_v", type=float, default=0.2)

    parser.add_argument("--use_contact", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=0)

    # dataset
    parser.add_argument("--past_len", type=int, default=10)
    parser.add_argument("--future_len", type=int, default=25)

    # train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_epoch", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)

    # diffusion
    parser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                        help="Noise schedule type")
    parser.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--cond_mask_prob", default=0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    parser.add_argument("--diverse_samples", type=int, default=10)
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    train_dataset = Dataset(mode = 'train', past_len=args.past_len, future_len=args.future_len)
    test_dataset = Dataset(mode = 'test', past_len=args.past_len, future_len=args.future_len)

    args.smpl_dim = 66 * 2
    args.num_obj_points = train_dataset.num_obj_points
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=False)  #pin_memory cause warning in pytorch 1.9.0
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')

    if args.resume_checkpoint is not None:
        print('resume training')
        model = LitInteraction.load_from_checkpoint(args.resume_checkpoint, args=args)
    else:
        print('start training from scratch')
        model = LitInteraction(args)

    if args.mode == "train":
        # callback
        tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/interaction'), name=args.expr_name)
        save_dir = Path(tb_logger.log_dir)  # for this version
        print(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                        monitor="val_loss",
                                                        save_weights_only=True, save_last=True)
        print(checkpoint_callback.dirpath)
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False,
                                                        mode="min")
        profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

        # trainer
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                profiler=profiler,
                                                # progress_bar_refresh_rate=1,
                                                callbacks=[checkpoint_callback, early_stop_callback],
                                                check_val_every_n_epoch=50,
                                                )
        trainer.fit(model, train_loader, val_loader)

    elif args.mode == "test" and args.resume_checkpoint is not None:
        # callback
        tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/sample'), name=args.expr_name)
        save_dir = Path(tb_logger.log_dir)  # for this version
        print(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                        monitor="val_loss",
                                                        save_weights_only=True, save_last=True)
        print(checkpoint_callback.dirpath)
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False,
                                                        mode="min")
        profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

        # trainer
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                profiler=profiler,
                                                # progress_bar_refresh_rate=1,
                                                callbacks=[checkpoint_callback, early_stop_callback],
                                                )
        trainer.test(model, val_loader)


