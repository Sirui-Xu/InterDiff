import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from data.dataset_smpl import Dataset, OBJECT_PATH
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
from render.mesh_viz import visualize_body_obj
from train_correction_smpl import LitInteraction as LitObj
from train_diffusion_smpl import LitInteraction
from data.utils import markerset_ssm67_smplh, SIMPLIFIED_MESH
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_quaternion
from copy import deepcopy
from tools import point2point_signed
from data.tools import vertex_normals

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def metrics(obj_pred, body_jtr, body, obj_gt, body_jtr_gt, body_gt, verts, faces, obj_points):
    # NOTE: could be modified for more efficient implementation
    # body_jtr, body_trans, obj_pred, body_jtr_gt, body_trans_gt, obj_gt
    # body_jtr: T, B, N_jtr, 3
    # body_trans: T, B, 3
    # obj_pred: T, B, 6  rot in first three items as axis-angle 
    # gender: from dataset, each example has a different gender
    T, B, N_jtr,_ = body_jtr_gt.shape# N_jtr 52?

    obj_rot_matrix = axis_angle_to_matrix(obj_pred[:, :, :-3].view(T, B, 3))
    obj_points_pred = torch.matmul(obj_points.unsqueeze(0), obj_rot_matrix.permute(0, 1, 3, 2)) + obj_pred[:, :, -3:].unsqueeze(2)

    normals = vertex_normals(verts.view(T * B, -1, 3), faces.unsqueeze(0).repeat(T * B, 1, 1))
    o2h_signed, h2o_signed, o2h_idx, h2o_idx, o2h, h2o = point2point_signed(verts.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=normals, return_vector=True)

    w_dist_neg = (o2h_signed < 0).view(T, B, -1).float()
    penetrate = w_dist_neg.mean(dim=2).mean(dim=0)

    body_trans = body[:, :, -3:]
    body_trans_gt = body_gt[:, :, -3:]
    # global mpjpe
    global_mpjpe = (body_jtr - body_jtr_gt).norm(dim=3).mean(dim=2).mean(dim=0)

    # align pelvis
    pelvis = body_jtr[:,:,0:1,:]
    pelvis_gt = body_jtr_gt[:,:,0:1,:]
    
    body_jtr = body_jtr - pelvis
    body_jtr_gt = body_jtr_gt - pelvis_gt
    # local mpjpe
    local_mpjpe = (body_jtr - body_jtr_gt).norm(dim=3).mean(dim=2).mean(dim=0)


    # body_translation
    body_translation = (body_trans - body_trans_gt).norm(dim=2).mean(dim=0)

    # translation
    obj_translation = (obj_pred[:,:,-3:] - obj_gt[:,:,-3:]).norm(dim=2).mean(dim=0)

    # quaternion error
    # to quat
    obj_rot_quat = axis_angle_to_quaternion(obj_pred[:,:,:3])# T,B,4
    obj_rot_quat_gt = axis_angle_to_quaternion(obj_gt[:,:,:3])# T,B,4

    rotation_error_v1 = (obj_rot_quat - obj_rot_quat_gt).norm(dim=2,p=1)
    rotation_error_v2 = (obj_rot_quat + obj_rot_quat_gt).norm(dim=2,p=1)
    rotation_min = torch.stack([rotation_error_v1, rotation_error_v2], dim=0).min(dim=0)[0]
    rotation_error = rotation_min.mean(dim=0)

    metric_dict = dict(
        global_mpjpe = global_mpjpe,
        local_mpjpe = local_mpjpe,
        body_translation = body_translation,
        obj_translation = obj_translation,
        obj_rot_error = rotation_error,
        penetrate = penetrate
    )
    return metric_dict


def denoised_fn(x, t, model_kwargs):
    if t[0] > 500 or t[0] % 50 != 0:
        return x
    body, obj = torch.split(x.squeeze(1).permute(2, 0, 1).contiguous(), args.smpl_dim+3, dim=2)
    body_gt, obj_gt = torch.split(model_kwargs['y']['inpainted_motion'].squeeze(1).permute(2, 0, 1).contiguous(), args.smpl_dim+3, dim=2)
    T, B, _ = body[:, :, :-3].shape
    obj_rot_matrix = rotation_6d_to_matrix(obj[:, :, :-3].view(T, B, 6))
    body_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
    hand_pose = model_kwargs['y']['hand_pose']
    body_pred = torch.cat([body_rot, hand_pose, body[:, :, -3:]], dim=2)

    body_pred = body_pred.detach().clone()     
    body_pred_batch = body_pred.view(T * B, -1)          
    smpl = model_kwargs['y']['smpl']
    betas_batch = model_kwargs['y']['beta'].view(T * B, -1)  
    verts, jtr, _, _ = smpl(body_pred_batch[:, :-3], 
                            th_betas=betas_batch, 
                            th_trans=body_pred_batch[:, -3:])
    markers = verts[:, markerset_ssm67_smplh]
    markers = markers.view(T, B, -1, 3)
    

    obj_model = model_kwargs['y']['obj_model']
    obj_points_pred = torch.matmul(model_kwargs['y']['obj_points'].unsqueeze(0), obj_rot_matrix.permute(0, 1, 3, 2)) + obj[:, :, -3:].unsqueeze(2)
    # print(torch.where((torch.norm((markers.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4) < 0.03).any(dim=3)))

    normals = vertex_normals(verts, smpl.th_faces.unsqueeze(0).repeat(T * B, 1, 1))
    o2h_signed, h2o_signed, o2h_idx, h2o_idx, o2h, h2o = point2point_signed(verts.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=normals, return_vector=True)

    w = torch.zeros([T * B, o2h_signed.size(1)]).to(o2h_signed.device)
    w_dist = (o2h_signed < 0.01) * (o2h_signed > 0)
    w_dist_neg = o2h_signed < 0
    w[w_dist] = 0 # small weight for far away vertices
    w[w_dist_neg] = 20 # large weight for penetration

    loss_dist_o = torch.einsum('ij,ij->ij', torch.abs(o2h_signed), w).view(T, B, -1) # 
    distance = (torch.norm((markers.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4)).min(dim=3)[0].min(dim=2)[0].mean(dim=0)
    condition = torch.logical_not(torch.logical_and(loss_dist_o[args.past_len:].mean(dim=2).mean(dim=0) < 0.002, distance < 0.02))
    contact_human_label = (torch.norm((markers.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4) < 0.02).any(dim=2)
    contact = torch.zeros_like(contact_human_label, device=contact_human_label.device)
    contact[contact_human_label] = 1
    contact = contact[args.past_len:].sum(dim=0)
    obj_proj = obj_model.model.sample(obj_gt[:, :, :-3], obj_gt[:, :, -3:], markers, contact)
    x_ = torch.cat([body, obj_proj], dim=2).permute(1, 2, 0).unsqueeze(1).contiguous()
    x_ = t[0] / 1000 * x + (1 - t[0] / 1000) * x_
    x[condition] = x_[condition]
    return x


def sample_once_proj(batch):
    with torch.no_grad():
        embedding, gt = model.model._get_embeddings(batch, device)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}}
        model_kwargs['y']['inpainted_motion'] = gt
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                device=device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, args.past_len:] = False  # do inpainting in those frames

        sample_fn = model.diffusion.p_sample_loop
        hand_pose = torch.cat([frame['smplfit_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float().to(device)
        model_kwargs['y']['hand_pose'] = hand_pose[idx_pad]
        smpl = model_kwargs['y']['smpl'] = model.body_model['male']
        model_kwargs['y']['beta'] = torch.stack([record['smplfit_params']['betas'] for record in batch['frames']], dim=0).to(device)
        model_kwargs['y']['obj_model'] = obj_model
        model_kwargs['y']['obj_points'] = batch['obj_points'][:, :, :3].float().to(device)

        noise = torch.randn(*gt.shape, device=device)
        sample = sample_fn(model.model, gt.shape, clip_denoised=False, noise=noise, model_kwargs=model_kwargs, denoised_fn=denoised_fn)
        body_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), args.smpl_dim+3, dim=2)
        body_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), args.smpl_dim+3, dim=2)
        T, B, _ = body_pred[:, :, :-3].shape
        body_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot_matrix = rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, 6))
        obj_rot = matrix_to_axis_angle(obj_rot_matrix).view(T, B, -1)
        obj_rot_gt_matrix = rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, 6))
        obj_rot_gt = matrix_to_axis_angle(obj_rot_gt_matrix).view(T, B, -1)
        body_pred = torch.cat([body_rot, hand_pose[idx_pad], body_pred[:, :, -3:]], dim=2)
        body_gt = torch.cat([body_rot_gt, hand_pose, body_gt[:, :, -3:]], dim=2)

        body = body_pred.detach().clone()     
        betas = torch.stack([record['smplfit_params']['betas'] for record in batch['frames']], dim=0).to(device) 
        body_batch = body.view(T * B, -1)          
        betas_batch = betas.view(T * B, -1)
        smpl = model.body_model['male']
        verts, jtr, _, _ = smpl(body_batch[:, :-3], 
                                th_betas=betas_batch, 
                                th_trans=body_batch[:, -3:])
        obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)


    return obj_pred, body_pred, verts.view(T, B, -1, 3), jtr.view(T, B, -1, 3), jtr.view(T, B, -1, 3)[:, :, 0, :]

def sample_once(batch):
    with torch.no_grad():
        embedding, gt = model.model._get_embeddings(batch, device)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}}
        model_kwargs['y']['inpainted_motion'] = gt
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                device=device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, args.past_len:] = False  # do inpainting in those frames

        sample_fn = model.diffusion.p_sample_loop

        noise = torch.randn(*gt.shape, device=device)
        sample = sample_fn(model.model, gt.shape, clip_denoised=False, noise=noise, model_kwargs=model_kwargs)
        body_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), args.smpl_dim+3, dim=2)
        body_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), args.smpl_dim+3, dim=2)
        T, B, _ = body_pred[:, :, :-3].shape
        body_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        hand_pose = torch.cat([frame['smplfit_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float().to(device) 
        body_pred = torch.cat([body_rot, hand_pose[idx_pad], body_pred[:, :, -3:]], dim=2)
        body_gt = torch.cat([body_rot_gt, hand_pose, body_gt[:, :, -3:]], dim=2)

        body = body_pred.detach().clone()     
        betas = torch.stack([record['smplfit_params']['betas'] for record in batch['frames']], dim=0).to(device) 
        body_batch = body.view(T * B, -1)     
        betas_batch = betas.view(T * B, -1)     
        smpl = model.body_model['male']
        verts, jtr, _, _ = smpl(body_batch[:, :-3], 
                                th_betas=betas_batch, 
                                th_trans=body_batch[:, -3:])
        obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)

    return obj_pred, body_pred, verts.view(T, B, -1, 3), jtr.view(T, B, -1, 3), jtr.view(T, B, -1, 3)[:, :, 0, :]

def smooth(obj, body, verts, jtrs, pelvis):
    obj[-args.future_len:] = obj[-args.future_len:] + (2 * obj[-args.future_len-1] - obj[-args.future_len-2] - obj[-args.future_len])
    body[-args.future_len:] = body[-args.future_len:] + (2 * body[-args.future_len-1] - body[-args.future_len-2] - body[-args.future_len])
    verts[-args.future_len:] = verts[-args.future_len:] + (2 * verts[-args.future_len-1] - verts[-args.future_len-2] - verts[-args.future_len])
    jtrs[-args.future_len:] = jtrs[-args.future_len:] + (2 * jtrs[-args.future_len-1] - jtrs[-args.future_len-2] - jtrs[-args.future_len])
    pelvis[-args.future_len:] = pelvis[-args.future_len:] + (2 * pelvis[-args.future_len-1] - pelvis[-args.future_len-2] - pelvis[-args.future_len])
    return obj, body, verts, jtrs, pelvis

def get_gt(batch):
    with torch.no_grad():
        embedding, gt = model.model._get_embeddings(batch, device)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()

        body_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), args.smpl_dim+3, dim=2)
        T, B, _ = body_gt[:, :, :-3].shape
        body_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        hand_pose = torch.cat([frame['smplfit_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float().to(device) 
        body_gt = torch.cat([body_rot_gt, hand_pose, body_gt[:, :, -3:]], dim=2)

        obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)

        smpl = model.body_model['male']
        faces = smpl.th_faces

        betas = torch.stack([record['smplfit_params']['betas'] for record in batch['frames']], dim=0).to(device) 
        betas_batch = betas.view(T * B, -1) 
        body_gt_batch = body_gt.view(T * B, -1)
        verts_gt, jtr_gt, _, _ = smpl(body_gt_batch[:, :-3], 
                th_betas=betas_batch, 
                th_trans=body_gt_batch[:, -3:])
        
    return obj_gt, jtr_gt.view(T, B, -1, 3), body_gt, faces

def sample(name):
    if name == 'correction':
        sample_func = sample_once_proj
    else:
        sample_func = sample_once

    metric_dict = dict(
        global_mpjpe = 0,
        local_mpjpe = 0,
        body_translation = 0,
        obj_translation = 0,
        obj_rot_error = 0,
        penetrate = 0
    )
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            global_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            local_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            body_translation = torch.zeros(args.batch_size).to(device) + 1e10
            obj_translation = torch.zeros(args.batch_size).to(device) + 1e10
            obj_rot_error = torch.zeros(args.batch_size).to(device) + 1e10
            penetrate = torch.zeros(args.batch_size).to(device) + 1e10
            obj_gt, jtr_gt, body_gt, faces = get_gt(batch)
            for j in range(args.diverse_samples):
                new_batch = deepcopy(batch)
                obj, body, verts, jtrs, pelvis = sample_func(new_batch)
                
                metric = metrics(obj[args.past_len:], jtrs[args.past_len:], body[args.past_len:], obj_gt[args.past_len:], jtr_gt[args.past_len:], body_gt[args.past_len:], verts[args.past_len:], faces, batch['obj_points'][:, :, :3].float().to(device))
                global_mpjpe = torch.stack([global_mpjpe, metric['global_mpjpe']])
                local_mpjpe = torch.stack([local_mpjpe, metric['local_mpjpe']])
                body_translation = torch.stack([body_translation, metric['body_translation']])
                obj_translation = torch.stack([obj_translation, metric['obj_translation']])
                obj_rot_error = torch.stack([obj_rot_error, metric['obj_rot_error']])
                penetrate = torch.stack([penetrate, metric['penetrate']])

                obj, body, verts, jtrs, pelvis = smooth(obj, body, verts, jtrs, pelvis)
                if i % args.render_epoch == 0:
                    visualize(batch, i, obj[:, 0], verts[:, 0], faces, name)

            metric_dict['global_mpjpe'] += global_mpjpe.min(dim=0)[0].mean().item()
            metric_dict['local_mpjpe'] += local_mpjpe.min(dim=0)[0].mean().item()
            metric_dict['body_translation'] += body_translation.min(dim=0)[0].mean().item()
            metric_dict['obj_translation'] += obj_translation.min(dim=0)[0].mean().item()
            metric_dict['obj_rot_error'] += obj_rot_error.min(dim=0)[0].mean().item()
            metric_dict['penetrate'] += penetrate.min(dim=0)[0].mean().item()
            print(i+1)
            print('global_mpjpe', metric_dict['global_mpjpe'] / (i+1))
            print('local_mpjpe', metric_dict['local_mpjpe'] / (i+1))
            print('body_translation', metric_dict['body_translation'] / (i+1))
            print('obj_translation', metric_dict['obj_translation'] / (i+1))
            print('obj_rot_error', metric_dict['obj_rot_error'] / (i+1))
            print('penetrate', metric_dict['penetrate'] / (i+1))
                        
def visualize(batch, j, obj, verts, faces, name):
    verts = verts.detach().cpu().numpy()
    faces = faces.cpu().numpy()
    obj_verts = []
    # visualize
    export_file = Path.joinpath(save_dir, 'render')
    export_file.mkdir(exist_ok=True, parents=True)
    # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
    rend_video_path = os.path.join(export_file, 's{}_l{}_r{}_{}_{}.gif'.format(batch['start_frame'][0], obj.shape[0], args.sample_rate, j, name))

    for t in range(obj.shape[0]):
        # print(record['smplfit_params'])
        mesh_obj = Mesh()
        mesh_obj.load_from_file(os.path.join(OBJECT_PATH, SIMPLIFIED_MESH[batch['obj_name'][0]]))
        mesh_obj_v = mesh_obj.v.copy()
        # center the meshes
        center = np.mean(mesh_obj_v, 0)
        mesh_obj_v = mesh_obj_v - center
        angle, trans = obj[t][:-3].detach().cpu().numpy(), obj[t][-3:].detach().cpu().numpy()
        rot = Rotation.from_rotvec(angle).as_matrix()
        # transform canonical mesh to fitting
        mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
        obj_verts.append(mesh_obj_v)

    m1 = visualize_body_obj(verts, faces, np.array(obj_verts), mesh_obj.f, past_len=args.past_len, save_path=rend_video_path, sample_rate=args.sample_rate)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='Diffusion')
    parser.add_argument("--use_pointnet2", type=int, default=1)
    parser.add_argument("--num_obj_keypoints", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=1)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0.1)
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
    parser.add_argument("--weight_past", type=float, default=0.5)
    parser.add_argument("--weight_jtr", type=float, default=0.1)
    parser.add_argument("--weight_jtr_v", type=float, default=500)
    parser.add_argument("--weight_v", type=float, default=1)

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
    parser.add_argument("--render_epoch", type=int, default=50)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--resume_checkpoint_obj", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--mode", type=str, default='correction')
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--dct", type=int, default=10)
    parser.add_argument("--autoregressive", type=int, default=0)

    # diffusion
    parser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                        help="Noise schedule type")
    parser.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--cond_mask_prob", default=0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    parser.add_argument("--diverse_samples", type=int, default=1)
    args = parser.parse_args()
    idx_pad = list(range(args.past_len)) + [args.past_len - 1] * args.future_len
    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    test_dataset = Dataset(mode = 'test', past_len=args.past_len, future_len=args.future_len)

    args.smpl_dim = 66 * 2
    args.num_obj_points = test_dataset.num_obj_points
    args.num_verts = len(markerset_ssm67_smplh)

    #pin_memory cause warning in pytorch 1.9.0
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')

    model = LitInteraction.load_from_checkpoint(args.resume_checkpoint, args=args).to(device)
    obj_model = LitObj.load_from_checkpoint(args.resume_checkpoint_obj, args=args).to(device)
    
    model.eval()
    obj_model.eval()
    tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/sample'), name=args.expr_name)
    save_dir = Path(tb_logger.log_dir)  # for this version
    print(save_dir)
    # sample()
    sample(args.mode)
