import os
from datetime import datetime
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from tools import point2point_signed
from model.correction_smpl import ObjProjector
from data.utils import SIMPLIFIED_MESH
from data.dataset_smpl import Dataset, OBJECT_PATH, MODEL_PATH
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
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

        self.model = ObjProjector(args)
        self.model.to(device=device, dtype=torch.float)

    def on_train_start(self) -> None:
        #     backup trainer and model
        shutil.copy('./train_correction_smpl.py', str(save_dir / 'train_correction_smpl.py'))
        shutil.copy('./model/correction_smpl.py', str(save_dir / 'correction_smpl.py'))
        shutil.copy('./data/dataset_smpl.py', str(save_dir / 'dataset_smpl.py'))
        return

    def forward(self, x, initialize=False):
        return self.model(x, initialize)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
        return ({'optimizer': optimizer,
                 })
    
    def calc_loss(self, obj_pred, obj_gt, batch):

        loss_dict = dict()
        weighted_loss_dict = {}

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

    def calc_loss_contact(self, obj_pred, obj_gt, batch):
        T, B, _ = obj_gt.shape

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

        obj_rot = rotation_6d_to_matrix(obj_pred[:, :, :-3]).permute(0, 1, 3, 2)
        obj_rot_gt = rotation_6d_to_matrix(obj_gt[:, :, :-3]).permute(0, 1, 3, 2)


        obj_points = batch['obj_points'].float() # (T)xBxPx6
        obj_points_pred = torch.matmul(obj_points.unsqueeze(0)[:, :, :, :3], obj_rot) + obj_pred[:, :, -3:].unsqueeze(2)

        human_verts = torch.cat([frame['human_verts'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBxPx7
        human_verts, human_verts_normals, human_contact_label = human_verts[:, :, :, :3], human_verts[:, :, :, 3:6], human_verts[:, :, :, 6:]

        o2h_signed, h2o_signed, o2h_idx, h2o_idx, o2h, h2o = point2point_signed(human_verts.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=human_verts_normals.view(T * B, -1, 3), return_vector=True)
        h2o = h2o.view(T, B, -1, 3)

        v_contact = torch.zeros([T * B, h2o_signed.size(1)]).to(h2o_signed.device)
        v_collision = torch.zeros([T * B, h2o_signed.size(1)]).to(h2o_signed.device)  
        v_dist = (torch.abs(h2o_signed) > 0.02) * (human_contact_label.view(T * B, -1) > 0.5) 

        v_contact[v_dist] = 1

        w = torch.zeros([T * B, o2h_signed.size(1)]).to(self.device)
        w_dist = (o2h_signed < 0.01) * (o2h_signed > 0)
        w_dist_neg = o2h_signed < 0
        w[w_dist] = 0 
        w[w_dist_neg] = 20
       
        f = torch.nn.ReLU()

        loss_contact = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed), v_contact))  
        loss_dist_o = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed), w)) 

        loss_contact = loss_contact  
    
        loss_penetration = loss_dist_o 

        loss_dict = dict(penetration=loss_penetration,
                         contact=loss_contact,
                         )
        annealing_factor = min(1.0, max(float(self.current_epoch) / (self.args.second_stage), 0)) if self.args.use_annealing else 1
        weighted_loss_dict = {
            'contact': max(annealing_factor ** 2, 0) * loss_contact * self.args.weight_contact,
            'penetration': max(annealing_factor ** 2, 0) * loss_penetration * self.args.weight_penetration,
        }

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
        obj_pred, obj_gt = self(batch, self.current_epoch < 10)
        loss, loss_dict, weighted_loss_dict = self.calc_loss_contact(obj_pred, obj_gt, batch=batch)

        render_interval = 50 if mode == 'valid' else 200
        if mode != 'train' and (batch_idx % render_interval == 0) and (((self.current_epoch+1) % self.args.render_epoch == 0) or self.args.debug):
            body_pose = torch.cat([frame['smplfit_params']['pose'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBxDb
            body_trans = torch.cat([frame['smplfit_params']['trans'].unsqueeze(0) for frame in batch['frames']], dim=0).float() # TxBx3
            body_gt = torch.cat([body_pose, body_trans], dim=2)
            T, B, _ = body_gt[:, :, :-3].shape
            obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, 6))).view(T, B, -1)
            obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, 6))).view(T, B, -1)
            obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)
            obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)

            with torch.no_grad():
                body = body_gt.detach().clone()
                obj = obj_pred.detach().clone()
                body_gt = body_gt.detach().clone()
                obj_gt = obj_gt.detach().clone()
                
                # visualize
                export_file = Path.joinpath(save_dir, 'render')
                export_file.mkdir(exist_ok=True, parents=True)
                # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
                rend_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))
                rend_gt_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_gt.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))
                rend_p_video_path = os.path.join(export_file, '{}_{}_{}_s{}_l{}_r{}_p.gif'.format(mode, self.current_epoch, batch_idx, batch['start_frame'][0], len(batch['frames'][0]), self.args.sample_rate))

                smpl = self.body_model[batch['gender'][0]]
                verts, jtr, _, _ = smpl(body[:, 0, :-3], 
                                        th_betas=torch.cat([record['smplfit_params']['betas'][0:1] for record in batch['frames']], dim=0), 
                                        th_trans=body[:, 0, -3:])
                jtr = jtr.detach().cpu().numpy()
                # print(np.argmin(jtr[:, :, 1], axis=1))
                verts = verts.detach().cpu().numpy()
                
                verts_gt, jtr_gt, _, _ = smpl(body_gt[:, 0, :-3], 
                                        th_betas=torch.cat([record['smplfit_params']['betas'][0:1] for record in batch['frames']], dim=0), 
                                        th_trans=body_gt[:, 0, -3:])
                jtr_gt = jtr_gt.detach().cpu().numpy()
                # print(np.argmin(jtr[:, :, 1], axis=1))
                verts_gt = verts_gt.detach().cpu().numpy()
                
                faces = smpl.th_faces.cpu().numpy()
                obj_verts = []
                obj_verts_gt = []
                for t, record in (enumerate(batch['frames'])):
                    # print(record['smplfit_params'])
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

                    # center the meshes
                    center = np.mean(mesh_obj_v, 0)
                    mesh_obj_v = mesh_obj_v - center
                    angle, trans = obj_gt[t][0][:-3].detach().cpu().numpy(), obj_gt[t][0][-3:].detach().cpu().numpy()
                    rot = Rotation.from_rotvec(angle).as_matrix()
                    # transform canonical mesh to fitting
                    mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
                    obj_verts_gt.append(mesh_obj_v)

                m1 = visualize_body_obj(np.concatenate((verts_gt[:args.past_len], verts[args.past_len:]), axis=0), faces, np.array(obj_verts_gt[:args.past_len] + obj_verts[args.past_len:]), mesh_obj.f, past_len=self.args.past_len, save_path=rend_video_path)
                m2 = visualize_body_obj(verts_gt, faces, np.array(obj_verts_gt), mesh_obj.f, past_len=self.args.past_len, save_path=rend_gt_video_path, sample_rate=self.args.sample_rate)
                m3 = visualize_body_obj(verts, faces, np.array(obj_verts), mesh_obj.f, past_len=self.args.past_len, save_path=rend_p_video_path, sample_rate=self.args.sample_rate)

        return loss, loss_dict, weighted_loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'train')

        self.log('train_loss', loss, prog_bar=False)
        for key in loss_dict:
            self.log(key, loss_dict[key], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='ObjProjector')
    parser.add_argument("--use_pointnet2", type=int, default=1)
    parser.add_argument("--num_obj_keypoints", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=1)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=256)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dct", type=int, default=10)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument('--star_graph', default=False, action='store_true')

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_template", type=float, default=0.1)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_contact", type=float, default=1)
    parser.add_argument("--weight_dist", type=float, default=0.1)
    parser.add_argument("--weight_penetration", type=float, default=0.1)  #10

    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.1)
    parser.add_argument("--weight_obj_rot", type=float, default=0.1)
    parser.add_argument("--weight_obj_nonrot", type=float, default=0.1)
    parser.add_argument("--weight_past", type=float, default=0.5)
    parser.add_argument("--weight_jtr", type=float, default=0.1)
    parser.add_argument("--weight_jtr_v", type=float, default=500)
    parser.add_argument("--weight_v", type=float, default=1)
    parser.add_argument("--use_contact", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=1)

    # dataset
    parser.add_argument("--past_len", type=int, default=10)
    parser.add_argument("--future_len", type=int, default=25)

    # train
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_epoch", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    train_dataset = Dataset(mode = 'train', past_len=args.past_len, future_len=args.future_len, sample_rate=args.sample_rate)
    test_dataset = Dataset(mode = 'test', past_len=args.past_len, future_len=args.future_len, sample_rate=args.sample_rate)

    args.smpl_dim = train_dataset.smpl_dim
    args.num_obj_points = train_dataset.num_obj_points
    args.num_verts = train_dataset.num_markers
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
                                            check_val_every_n_epoch=25,
                                            )
    trainer.fit(model, train_loader, val_loader)



