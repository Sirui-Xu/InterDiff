import os
from datetime import datetime
import shutil
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from model.diffusion_skeleton import create_model_and_diffusion
import functools
from diffusion.resample import LossAwareSampler
from diffusion.resample import create_named_schedule_sampler

from render.viz_helper import visualize_skeleton
from data.dataset_skeleton import get_datasets
from pytorch3d.transforms import quaternion_to_matrix

class LitInteraction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
        self.test_time= int(0)

        self.model, self.diffusion = create_model_and_diffusion(args)
        self.use_ddp = False
        self.ddp_model = self.model
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

    def on_train_start(self) -> None:
        #     backup trainer and model
        shutil.copy('./train_diffusion_skeleton.py', str(save_dir / 'train_diffusion_skeleton.py'))
        shutil.copy('./model/diffusion_skeleton.py', str(save_dir / 'diffusion_skeleton.py'))
        shutil.copy('./data/dataset_skeleton.py', str(save_dir / 'dataset_skeleton.py'))
        shutil.copy('./diffusion/gaussian_diffusion.py', str(save_dir / 'gaussian_diffusion.py'))
        return

    def l2(self, a, b):
        # assuming a.shape == b.shape == seqlen, bs, N
        loss = torch.nn.MSELoss(reduction='none')(a, b)
        loss = loss.mean(dim=[0, 2]) 
        return loss

    def calc_obj_pred(self, pose_pred, zero_pose_obj):
        # input: pose pred T,B,7
        # zero_pose_obj: B,N_points,3
        # return obj_pred: T,B,N_points,3
        # quaternion to matrix
        obj_gt_base = zero_pose_obj[None,:,:,:,None] # 1,B,N_points,3,1
        translation = pose_pred[:,:,None,:3,None] # T,B,1,3,1
        quat_correct = torch.cat([pose_pred[:,:,-1,None], pose_pred[:,:,-4:-1]],dim=2)
        rotation_matrix = quaternion_to_matrix(quat_correct)[:,:,None]# T,B,1,3,3
        obj_pred = rotation_matrix.matmul(obj_gt_base) + translation

        return obj_pred[:,:,:,:,0]

    def calc_metric(self, body_pred, body_gt, obj_pred, obj_gt, pose_pred, pose_gt):
        # body_pred: T,B,N_joints,3
        # obj_pred: T,B,N_points,3
        # pose_pred: T,B,7

        assert body_pred.size()[-1]==3
        assert obj_pred.size()[-1]==3

        mpjpe_h = (body_pred[10:] - body_gt[10:]).norm(dim=-1,p=2).mean().item()
        mpjpe_o = (obj_pred[10:] - obj_gt[10:]).norm(dim=-1,p=2).mean().item()
        translation_error = (pose_pred[10:,:,:3] - pose_gt[10:,:,:3]).norm(dim=-1,p=2).mean().item()

        # we have to modify
        rotation_error_v1 = (pose_pred[10:,:,-4:] - pose_gt[10:,:,-4:]).norm(dim=-1,p=2)
        rotation_error_v2 = (pose_pred[10:,:,-4:] + pose_gt[10:,:,-4:]).norm(dim=-1,p=2)
        rotation_min = torch.stack([rotation_error_v1, rotation_error_v2], dim=0).min(dim=0)[0]
        rotation_error = rotation_min.mean().item()
        metric_dict = dict(mpjpe_h=mpjpe_h,
        mpjpe_o = mpjpe_o,
        translation_error = translation_error,
        rotation_error = rotation_error
        )
        return metric_dict

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
        body_pred, obj_pred, pose_pred= torch.split(pred.squeeze(1).permute(2, 0, 1).contiguous(), 
         [self.args.num_joints*3, self.args.num_points*3,7], dim=2)
        body_gt, obj_gt, pose_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), 
         [self.args.num_joints*3, self.args.num_points*3,7], dim=2)

        # NOTE: check losses
        T,B = body_pred.size()[:2]
        loss_body_past = torch.nn.MSELoss(reduction='mean')(body_pred[:self.args.past_len], body_gt[:self.args.past_len])
        loss_body_future  = torch.nn.MSELoss(reduction='mean')(body_pred[self.args.past_len:], body_gt[self.args.past_len:])

        loss_obj_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:self.args.past_len], obj_gt[:self.args.past_len])
        loss_obj_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:], obj_gt[self.args.past_len:])

        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(pose_pred[:self.args.past_len,:,:3], pose_gt[:self.args.past_len,:,:3]) 
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(pose_pred[self.args.past_len:,:,:3], pose_gt[self.args.past_len:,:,:3])
        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(pose_pred[:self.args.past_len,:,-4:], pose_gt[:self.args.past_len,:,-4:]) 
        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(pose_pred[self.args.past_len:,:,-4:], pose_gt[self.args.past_len:,:,-4:])
        
        # add velocity
        loss_obj_nonrot_v = torch.nn.MSELoss(reduction='mean')(pose_pred[1:,:,:3] - pose_pred[:-1,:,:3], \
        pose_gt[1:,:,:3] - pose_gt[:-1,:,:3])
        loss_obj_rot_v = torch.nn.MSELoss(reduction='mean')(pose_pred[1:,:,-4:] - pose_pred[:-1,:,-4:], \
        pose_gt[1:,:,-4:] - pose_gt[:-1,:,-4:])
        loss_obj_v = torch.nn.MSELoss(reduction='mean')(obj_pred[1:] - obj_pred[:-1], obj_gt[1:] - obj_gt[:-1])
        loss_body_v = torch.nn.MSELoss(reduction='mean')(body_pred[1:] - body_pred[:-1], body_gt[1:] - body_gt[:-1])
        # quaternion normalization in QuaterNet
        quaternion_reg_loss = (pose_pred[:,:,-4:].norm(p=2, dim=-1).square()-1).square().mean()

        loss_dict = dict(
                         body_past = loss_body_past,
                         body_future = loss_body_future,
                         obj_past = loss_obj_past,
                         obj_future = loss_obj_future,
                         loss_obj_nonrot_past = loss_obj_nonrot_past,
                         loss_obj_nonrot_future = loss_obj_nonrot_future,
                         loss_obj_rot_past = loss_obj_rot_past,
                         loss_obj_rot_future = loss_obj_rot_future,
                         quaternion_reg_loss = quaternion_reg_loss,
                         loss_obj_rot_v = loss_obj_rot_v,
                         loss_obj_nonrot_v = loss_obj_nonrot_v,
                         loss_body_v = loss_body_v,
                         loss_obj_v = loss_obj_v
                         )

        weighted_loss_dict = dict(
                         body_past = loss_body_past * self.args.weight_body * self.args.weight_past,
                         body_future = loss_body_future * self.args.weight_body,
                         obj_past = loss_obj_past * self.args.weight_obj* self.args.weight_past,
                         obj_future = loss_obj_future * self.args.weight_obj,
                         loss_obj_nonrot_past = loss_obj_nonrot_past * self.args.weight_obj_nonrot* self.args.weight_past,
                         loss_obj_nonrot_future = loss_obj_nonrot_future* self.args.weight_obj_nonrot,
                         loss_obj_rot_past = loss_obj_rot_past* self.args.weight_obj_rot* self.args.weight_past,
                         loss_obj_rot_future = loss_obj_rot_future* self.args.weight_obj_rot,
                         quaternion_reg_loss = self.args.weight_quat_reg* quaternion_reg_loss,
                         loss_obj_rot_v = loss_obj_rot_v*self.args.weight_obj_rot*self.args.weight_v,
                         loss_obj_nonrot_v = loss_obj_nonrot_v*self.args.weight_obj_nonrot*self.args.weight_v,
                         loss_body_v = loss_body_v* self.args.weight_body*self.args.weight_v ,
                         loss_obj_v = loss_obj_v* self.args.weight_obj*self.args.weight_v
                         )

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, loss.detach()
            )

        loss = (loss * weights).mean()
        self.log_loss_dict(
            self.diffusion, t, weighted_loss_dict, loss
        )
        metric_dict = self.calc_metric(body_pred.view(T,B,self.args.num_joints,3), body_gt.view(T,B,self.args.num_joints,3),\
         obj_pred.view(T,B,self.args.num_points,3), obj_gt.view(T,B,self.args.num_points,3), pose_pred, pose_gt)

        return loss, body_pred, obj_pred, body_gt, obj_gt

    def log_loss_dict(self, diffusion, ts, losses, loss):
        self.log('train_loss', loss, prog_bar=False)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=list(self.model.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
        return ({'optimizer': optimizer,
                 })

    def calc_val_loss(self, body_pred, body_gt, obj_pred, obj_gt, pose_pred, pose_gt, batch):
        T,B = body_pred.size()[:2]
        loss_body_past = torch.nn.MSELoss(reduction='mean')(body_pred[:self.args.past_len], body_gt[:self.args.past_len])
        loss_body_future  = torch.nn.MSELoss(reduction='mean')(body_pred[self.args.past_len:], body_gt[self.args.past_len:])

        loss_obj_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:self.args.past_len], obj_gt[:self.args.past_len])
        loss_obj_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:], obj_gt[self.args.past_len:])

        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(pose_pred[:self.args.past_len,:,:3], pose_gt[:self.args.past_len,:,:3]) 
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(pose_pred[self.args.past_len:,:,:3], pose_gt[self.args.past_len:,:,:3])
        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(pose_pred[:self.args.past_len,:,-4:], pose_gt[:self.args.past_len,:,-4:]) 
        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(pose_pred[self.args.past_len:,:,-4:], pose_gt[self.args.past_len:,:,-4:])
        
        # quaternion normalization in QuaterNet
        quaternion_reg_loss = (pose_pred[:,:,-4:].norm(p=2, dim=-1).square()-1).square().mean()

        # add v loss
        # NOTE: add velocity
        loss_obj_nonrot_v = torch.nn.MSELoss(reduction='mean')(pose_pred[1:,:,:3] - pose_pred[:-1,:,:3], \
        pose_gt[1:,:,:3] - pose_gt[:-1,:,:3])
        loss_obj_rot_v = torch.nn.MSELoss(reduction='mean')(pose_pred[1:,:,-4:] - pose_pred[:-1,:,-4:], \
        pose_gt[1:,:,-4:] - pose_gt[:-1,:,-4:])
        loss_obj_v = torch.nn.MSELoss(reduction='mean')(obj_pred[1:] - obj_pred[:-1], obj_gt[1:] - obj_gt[:-1])
        loss_body_v = torch.nn.MSELoss(reduction='mean')(body_pred[1:] - body_pred[:-1], body_gt[1:] - body_gt[:-1])

        loss_dict = dict(
                         body_past = loss_body_past,
                         body_future = loss_body_future,
                         obj_past = loss_obj_past,
                         obj_future = loss_obj_future,
                         loss_obj_nonrot_past = loss_obj_nonrot_past,
                         loss_obj_nonrot_future = loss_obj_nonrot_future,
                         loss_obj_rot_past = loss_obj_rot_past,
                         loss_obj_rot_future = loss_obj_rot_future,
                         quaternion_reg_loss = quaternion_reg_loss,
                         loss_obj_rot_v = loss_obj_rot_v,
                         loss_obj_nonrot_v = loss_obj_nonrot_v,
                         loss_body_v = loss_body_v,
                         loss_obj_v = loss_obj_v
                         )

        weighted_loss_dict = dict(
                         body_past = loss_body_past * self.args.weight_body * self.args.weight_past,
                         body_future = loss_body_future * self.args.weight_body,
                         obj_past = loss_obj_past * self.args.weight_obj* self.args.weight_past,
                         obj_future = loss_obj_future * self.args.weight_obj,
                         loss_obj_nonrot_past = loss_obj_nonrot_past * self.args.weight_obj_nonrot* self.args.weight_past,
                         loss_obj_nonrot_future = loss_obj_nonrot_future* self.args.weight_obj_nonrot,
                         loss_obj_rot_past = loss_obj_rot_past* self.args.weight_obj_rot* self.args.weight_past,
                         loss_obj_rot_future = loss_obj_rot_future* self.args.weight_obj_rot,
                         quaternion_reg_loss = self.args.weight_quat_reg* quaternion_reg_loss,
                         loss_obj_rot_v = loss_obj_rot_v*self.args.weight_obj_rot*self.args.weight_v,
                         loss_obj_nonrot_v = loss_obj_nonrot_v*self.args.weight_obj_nonrot*self.args.weight_v,
                         loss_body_v = loss_body_v* self.args.weight_body*self.args.weight_v ,
                         loss_obj_v = loss_obj_v* self.args.weight_obj*self.args.weight_v
                         )

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        metric_dict = self.calc_metric(body_pred.view(T,B,self.args.num_joints,3), body_gt.view(T,B,self.args.num_joints,3),\
         obj_pred.view(T,B,self.args.num_points,3), obj_gt.view(T,B,self.args.num_points,3), pose_pred, pose_gt)


        return loss, loss_dict, weighted_loss_dict

    def _common_step(self, batch, batch_idx, mode):
        body_gt = batch[0].transpose(0,1).float()
        obj_gt = batch[1].transpose(0,1).float()
        pose_gt = batch[2].transpose(0,1).float()
        zero_pose_obj = batch[3].float() # B,N_points,3

        embedding, gt = self.model._get_embeddings(body_gt, obj_gt, pose_gt, zero_pose_obj)
        # T,B,106
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}, 'zero_pose_obj': zero_pose_obj}
        if mode == 'train':
            loss, body_pred, obj_pred, body_gt, obj_gt = self.forward_backward(gt, model_kwargs)
            render_interval = 200
            if self.args.render and (batch_idx % render_interval == 0) :
                self.visualize(body_pred, obj_pred, body_gt, obj_gt, batch, batch_idx, mode)
            return loss
        elif mode == 'valid' or mode=='test':
            # gt: B,1,N_gt,T
            model_kwargs['y']['inpainted_motion'] = gt
            model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                    device=device)  # True means use gt motion
            model_kwargs['y']['inpainting_mask'][:, :, :, self.args.past_len:] = False  # do inpainting in those frames
            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(self.model, gt.shape, clip_denoised=False, model_kwargs=model_kwargs)
            body_pred, obj_pred, pose_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(),
             [self.args.num_joints*3, self.args.num_points*3,7], dim=2)
            body_gt, obj_gt, pose_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), 
            [self.args.num_joints*3, self.args.num_points*3,7], dim=2)
            
            loss, loss_dict, weighted_loss_dict = self.calc_val_loss(body_pred, body_gt, obj_pred, obj_gt, pose_pred, pose_gt, batch=batch)
            
            render_interval = 50
            if self.args.render and (batch_idx % render_interval == 0) and (((self.current_epoch % self.args.render_epoch) == self.args.render_epoch-1) or self.args.debug):
                self.visualize(body_pred, obj_pred, body_gt, obj_gt, batch, batch_idx, mode)
                export_file = Path.joinpath(save_dir, 'val_sample')
                export_file.mkdir(exist_ok=True, parents=True)
                save_val_path = os.path.join(export_file, '{}_{}_{}_sample.pt'.format(mode, self.current_epoch, batch_idx))
                torch.save((body_pred[:,0], obj_pred[:,0], pose_pred[:,0]),save_val_path)
                # save file
            return loss, loss_dict, weighted_loss_dict


    def visualize(self, body_pred, obj_pred, body_gt, obj_gt, batch, batch_idx, mode, sample_idx=None):
        # NOTE: rewrite visualization
        # assuming same input shape
        T,B = body_pred.shape[:2]
        with torch.no_grad():
            skeledonData = body_pred[:,0].cpu().numpy().reshape(T,21, 3)            
            objData = obj_pred[:,0].cpu().numpy().reshape(T,12, 3)
            skeledonData_gt = body_gt[:,0].cpu().numpy().reshape(T,21, 3)            
            objData_gt = obj_gt[:,0].cpu().numpy().reshape(T,12, 3)

            export_file = Path.joinpath(save_dir, 'render')
            export_file.mkdir(exist_ok=True, parents=True)
            if sample_idx is not None:
                rend_video_path = os.path.join(export_file, '{}_{}_{}_{}_p.gif'.format(mode, self.current_epoch, batch_idx, sample_idx))
            else:
                rend_video_path = os.path.join(export_file, '{}_{}_{}_p.gif'.format(mode, self.current_epoch, batch_idx))

            visualize_skeleton(skeledonData, objData, save_dir = rend_video_path)
            if sample_idx is not None:
                rend_video_path = os.path.join(export_file, '{}_{}_{}_{}_gt.gif'.format(mode, self.current_epoch, batch_idx, sample_idx))
            else:
                rend_video_path = os.path.join(export_file, '{}_{}_{}_gt.gif'.format(mode, self.current_epoch, batch_idx))
            if sample_idx is not None and sample_idx>0: # no need to render gt more than once
                return 
            visualize_skeleton(skeledonData_gt, objData_gt, save_dir = rend_video_path)

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        if batch_idx==0 and self.test_time==1:
            print("switching to test 2")
            self.test_time=int(2)
        elif batch_idx==0 and self.test_time==0:
            self.test_time=int(1)      
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'test')

        for key in loss_dict:
            self.log('test_' + key, loss_dict[key], prog_bar=False)
        self.log('test_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--model", type=str, default='Diffusion')
    parser.add_argument("--num_joints", type=int, default=21)
    parser.add_argument("--num_points", type=int, default=12)

    # transformer
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=256)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--latent_usage", type=str, default='memory')

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)

    parser.add_argument("--weight_past", type=float, default=0.5)
    parser.add_argument("--weight_body", type=float, default=2)
    parser.add_argument("--weight_obj", type=float, default=1)
    parser.add_argument("--weight_obj_rot", type=float, default=1)
    parser.add_argument("--weight_obj_nonrot", type=float, default=1)

    parser.add_argument("--weight_quat_reg", type=float, default=0.01)
    parser.add_argument("--weight_v", type=float, default=1)

    # dataset
    parser.add_argument("--past_len", type=int, default=10)
    parser.add_argument("--future_len", type=int, default=25)
    parser.add_argument("--align_data", default=False, action='store_true')
    parser.add_argument("--discard_discrep", default=False, action='store_true')

    # train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=200)

    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%b-%j-%H:%M"))
    parser.add_argument("--render_epoch", type=int, default=5)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--render", default=False, action='store_true')

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

    args.smpl_dim = args.num_joints*3

    train_set, val_set, test_set, unseen_test_set = get_datasets(args.align_data, args.discard_discrep)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1, shuffle=True,
                            drop_last=True, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=1, shuffle=True,
                    drop_last=True, pin_memory=False)
    test_loader_unseen = DataLoader(unseen_test_set, batch_size=args.batch_size, num_workers=1, shuffle=False,
                    drop_last=True, pin_memory=False)
    test_loader_seen = DataLoader(test_set, batch_size=args.batch_size, num_workers=1, shuffle=False,
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
        tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/interaction_diffusion_v2'), name=args.expr_name)
        save_dir = Path(tb_logger.log_dir)  # for this version
        print(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                        monitor="val_loss",
                                                        save_weights_only=True, save_last=True)
        print(checkpoint_callback.dirpath)
        profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

        # trainer
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                profiler=profiler,
                                                # progress_bar_refresh_rate=1,
                                                callbacks=[checkpoint_callback],
                                                check_val_every_n_epoch=10,
                                                )
        trainer.fit(model, train_loader, val_loader)

