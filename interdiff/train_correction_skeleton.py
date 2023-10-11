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
from model.correction_skeleton import ObjProjector
from viz.viz_helper import visualize_skeleton
from dataset.dataset import get_datasets

from pytorch3d.transforms import quaternion_to_matrix

class LitObjInteraction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")

        self.model = ObjProjector(args)
        self.model.to(device=device, dtype=torch.float)

    def on_train_start(self) -> None:
        #     backup trainer and model
        shutil.copy('./train_correction_skeleton.py', str(save_dir / 'train_correction_skeleton.py'))
        shutil.copy('./model/correction_skeleton.py', str(save_dir / 'correction_skeleton.py'))
        shutil.copy('./data/dataset_skeleton.py', str(save_dir / 'dataset_skeleton.py'))
        return

    # def forward(self, x):
    #     return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        return ({'optimizer': optimizer,
                 })

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
        rotation_error_v1 = (pose_pred[10:,:,-4:] - pose_gt[10:,:,-4:]).norm(dim=-1,p=1)
        rotation_error_v2 = (pose_pred[10:,:,-4:] + pose_gt[10:,:,-4:]).norm(dim=-1,p=1)
        rotation_min = torch.stack([rotation_error_v1, rotation_error_v2], dim=0).min(dim=0)[0]
        rotation_error = rotation_min.mean().item()
        metric_dict = dict(mpjpe_h=mpjpe_h,
        mpjpe_o = mpjpe_o,
        translation_error = translation_error,
        rotation_error = rotation_error
        )
        return metric_dict
    
    def calc_loss(self, obj_pred, obj_gt):
        T, B, _ = obj_gt.shape
        # use quaternions here, but 6d inside projector
        loss_dict = dict()
        weighted_loss_dict = dict()
        obj_rot = obj_pred[:, :, :-3]
        obj_rot_gt = obj_gt[:, :, :-3]

        loss_obj_rot_past = torch.nn.MSELoss(reduction='mean')(obj_rot[:self.args.past_len], obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_past = torch.nn.MSELoss(reduction='mean')(obj_pred[:self.args.past_len, :, -3:], obj_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_v_past = torch.nn.MSELoss(reduction='mean')(obj_rot[1:self.args.past_len+1]-obj_rot[:self.args.past_len], obj_rot_gt[1:self.args.past_len+1]-obj_rot_gt[:self.args.past_len])
        loss_obj_nonrot_v_past = torch.nn.MSELoss(reduction='mean')(obj_pred[1:self.args.past_len+1, :, -3:]-obj_pred[:self.args.past_len, :, -3:], obj_gt[1:self.args.past_len+1, :, -3:]-obj_gt[:self.args.past_len, :, -3:])

        loss_obj_rot_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:], obj_rot_gt[self.args.past_len:])
        loss_obj_nonrot_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:], obj_gt[self.args.past_len:, :, -3:])

        loss_obj_rot_v_future = torch.nn.MSELoss(reduction='mean')(obj_rot[self.args.past_len:]-obj_rot[self.args.past_len-1:-1], obj_rot_gt[self.args.past_len:]-obj_rot_gt[self.args.past_len-1:-1])
        loss_obj_nonrot_v_future = torch.nn.MSELoss(reduction='mean')(obj_pred[self.args.past_len:, :, -3:]-obj_pred[self.args.past_len-1:-1, :, -3:], obj_gt[self.args.past_len:, :, -3:]-obj_gt[self.args.past_len-1:-1, :, -3:])

        loss_dict.update(dict(obj_rot_past=loss_obj_rot_past,
                              obj_nonrot_past=loss_obj_nonrot_past,
                              obj_rot_future=loss_obj_rot_future,
                              obj_nonrot_future=loss_obj_nonrot_future,
                              obj_rot_v_past=loss_obj_rot_v_past,
                              obj_nonrot_v_past=loss_obj_nonrot_v_past,
                              obj_rot_v_future=loss_obj_rot_v_future,
                              obj_nonrot_v_future=loss_obj_nonrot_v_future,
                             ))
        weighted_loss_dict.update(dict(obj_rot_past=loss_obj_rot_past * self.args.weight_obj_rot * self.args.weight_past,
                                       obj_nonrot_past=loss_obj_nonrot_past * self.args.weight_obj_nonrot * self.args.weight_past,
                                       obj_rot_future=loss_obj_rot_future * self.args.weight_obj_rot,
                                       obj_nonrot_future=loss_obj_nonrot_future * self.args.weight_obj_nonrot,
                                       obj_rot_v_past=loss_obj_rot_v_past * self.args.weight_v * self.args.weight_obj_rot * self.args.weight_past,
                                       obj_nonrot_v_past=loss_obj_nonrot_v_past * self.args.weight_v * self.args.weight_obj_nonrot * self.args.weight_past,
                                       obj_rot_v_future=loss_obj_rot_v_future * self.args.weight_v * self.args.weight_obj_rot,
                                       obj_nonrot_v_future=loss_obj_nonrot_v_future * self.args.weight_v * self.args.weight_obj_nonrot,
                                      ))

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def _common_step(self, batch, batch_idx, mode):
        body_gt = batch[0].transpose(0,1).float()#T,B,N_joints,3
        obj_gt = batch[1].transpose(0,1).float()
        pose_gt = batch[2].transpose(0,1).float()
        zero_pose_obj = batch[3].float() # B,N_points,3
        obj_trans, obj_angles = torch.split(pose_gt, [3,4], dim=2)
        T, B = body_gt.shape[:2]
        # T,B,4
        # T,B,3
        obj_angles_p, obj_trans_p, obj_angles_gt, obj_trans_gt = self.model(obj_angles, obj_trans, body_gt)
        pose_pred = torch.cat([obj_trans_p, obj_angles_p], dim=2)
        assert not pose_pred.isnan().any()
        assert (torch.cat([obj_trans_gt, obj_angles_gt], dim=2) - pose_gt).norm()<1e-4

        obj_pred = self.calc_obj_pred(pose_pred, zero_pose_obj)

        loss, loss_dict, weighted_loss_dict = self.calc_loss(pose_pred, pose_gt)

        render_interval = 50 if mode == 'valid' else 200
        if self.args.render and mode != 'train' and (batch_idx % render_interval == 0) and (((self.current_epoch+1) % self.args.render_epoch == 0) or self.args.debug):
            self.visualize(body_gt, obj_pred, body_gt, obj_gt, batch, batch_idx, mode)

        metric_dict = self.calc_metric(body_gt.view(T,B,self.args.num_joints,3), body_gt.view(T,B,self.args.num_joints,3),\
        obj_pred.view(T,B,self.args.num_points,3), obj_gt.view(T,B,self.args.num_points,3), pose_pred, pose_gt)
        # for key, values in metric_dict.items():
        #     self.log(mode + '_'+key, values, prog_bar=True)
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
            if sample_idx is not None and sample_idx>0: # no need to render gt many times
                return 
            visualize_skeleton(skeledonData_gt, objData_gt, save_dir = rend_video_path)

    def training_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'train')

        self.log('train_loss', loss, prog_bar=False)
        # for key in loss_dict:
        #     self.log(key, loss_dict[key], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        # for key in loss_dict:
        #     self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'test')

        for key in loss_dict:
            self.log('test_' + key, loss_dict[key], prog_bar=False)
        self.log('test_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser(description="obj")
    parser.add_argument("--model", type=str, default='ObjProjector')
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--num_joints", type=int, default=21)
    parser.add_argument("--num_points", type=int, default=12)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)


    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.1)
    parser.add_argument("--weight_obj_rot", type=float, default=0.1)
    parser.add_argument("--weight_obj_nonrot", type=float, default=0.1)
    parser.add_argument("--weight_past", type=float, default=0.5)
    parser.add_argument("--weight_v", type=float, default=1)


    # dataset
    parser.add_argument("--past_len", type=int, default=10)
    parser.add_argument("--future_len", type=int, default=10)
    parser.add_argument("--align_data", default=False, action='store_true')
    parser.add_argument("--discard_discrep", default=False, action='store_true')
    # train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=300)

    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_epoch", type=int, default=1)
    parser.add_argument("--render", default=False, action='store_true')
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    # NOTE: load the same dataset
    train_set, val_set, test_set, unseen_test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1, shuffle=True,
                            drop_last=True, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=1, shuffle=True,
                    drop_last=True, pin_memory=False)
    test_loader_unseen = DataLoader(unseen_test_set, batch_size=args.batch_size, num_workers=1, shuffle=False,
                    drop_last=False, pin_memory=False)
    test_loader_seen = DataLoader(test_set, batch_size=args.batch_size, num_workers=1, shuffle=False,
                    drop_last=False, pin_memory=False)  
    print('dataset loaded')

    if args.resume_checkpoint is not None:
        print('resume training')
        model = LitObjInteraction.load_from_checkpoint(args.resume_checkpoint, args=args)
    else:
        print('start training from scratch')
        model = LitObjInteraction(args)

    # callback
    tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/interaction_obj'), name=args.expr_name)
    save_dir = Path(tb_logger.log_dir)  # for this version
    print(save_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                       every_n_epochs=40 ,
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
                                            check_val_every_n_epoch=5,
                                            )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(ckpt_path="best", dataloaders=test_loader_seen)
    # trainer.test(ckpt_path="best", dataloaders=test_loader_unseen)
    


