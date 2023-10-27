'''test our model'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from train_diffusion_skeleton import LitInteraction
from data.dataset_skeleton import get_datasets
from pytorch3d.transforms import quaternion_to_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def body_obj_to_contact(body, obj):
    # find closest point and check whether less than threshold 0.1meter
    # input
    # T,B,N_joints,3
    # T,B,N_points,3
    # output: T,B,N_joints
    T, B, N_joints = body.shape[:3]
    contact = torch.zeros((T, B, N_joints)).to(body.device)
    distance_matrix = (body[:,:,:,None] -  obj[:,:,None]).norm(dim=4) # T,B,N_joints,N_points
    min_distance = distance_matrix.min(dim=3)[0]#T,B,N_joints
    min_indices = min_distance.argmin(dim=2, keepdim=True)#T,B,1
    # set corresponding positions to 1
    for i in range(T):
        for j in range(B):
            contact[i,j,min_indices[i,j]] = 1 if min_distance[i,j][min_indices[i,j]]<0.1 else 0
    return contact

def calc_obj_pred(pose_pred, zero_pose_obj):
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

def calc_metric_single(body_pred, body_gt, obj_pred, obj_gt, pose_pred, pose_gt):
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

def get_batch(old_batch, body_pred, obj_pred, pose_pred):
    batch = [None, None, None, old_batch[3]]
    T,B,N_joints,_ = body_pred.shape
    N_points = obj_pred.shape[2]
    body_pred = torch.cat([body_pred[-10:], body_pred[-1].unsqueeze(0).repeat(T-10,1,1,1)], dim=0)
    obj_pred = torch.cat([obj_pred[-10:], obj_pred[-1].unsqueeze(0).repeat(T-10,1,1,1)], dim=0)
    pose_pred = torch.cat([pose_pred[-10:], pose_pred[-1].unsqueeze(0).repeat(T-10,1,1)], dim=0)
    batch[0] = body_pred.transpose(0,1)
    batch[1] = obj_pred.transpose(0,1)
    batch[2] = pose_pred.transpose(0,1)
    return batch

def denoised_fn(x, t, model_kwargs):
    return x

def sample_once_proj(batch):
    # NOTE: should return sth for prediction and visualization
    with torch.no_grad():
        body_gt = batch[0].transpose(0,1).float().to(device)
        obj_gt = batch[1].transpose(0,1).float().to(device)
        pose_gt = batch[2].transpose(0,1).float().to(device)
        zero_pose_obj = batch[3].float().to(device) # B,N_points,3

        embedding, gt = model.model._get_embeddings(body_gt, obj_gt, pose_gt, zero_pose_obj)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()

        model_kwargs = {'y': {'cond': embedding}, 'zero_pose_obj': zero_pose_obj}
        model_kwargs['y']['inpainted_motion'] = gt
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                device=device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, args_diffusion.past_len:] = False  # do inpainting in those frames

        sample_fn = model.diffusion.p_sample_loop

        noise = torch.randn(*gt.shape, device=device)
        sample = sample_fn(model.model, gt.shape, clip_denoised=False, noise=noise, model_kwargs=model_kwargs, denoised_fn=denoised_fn)
        body_pred, obj_pred, pose_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(),
            [args_diffusion.num_joints*3, args_diffusion.num_points*3,7], dim=2)
        body_gt, obj_gt, pose_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), 
        [args_diffusion.num_joints*3, args_diffusion.num_points*3,7], dim=2)

    return obj_pred, body_pred, pose_pred, obj_gt, body_gt, pose_gt 


def sample(t, dataloader, data_size):
    sample_func = sample_once_proj
    metric_dict_batch = dict(mpjpe_h=[],
    mpjpe_o = [],
    translation_error = [],
    rotation_error = []
    )
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            obj_pred, body_pred, pose_pred, obj_gt, body_gt, pose_gt = sample_func(batch)
            T,B = pose_gt.shape[:2]
            metric_dict = calc_metric_single(body_pred.view(T,B,-1,3), body_gt.view(T,B,-1,3), obj_pred.view(T,B,-1,3), obj_gt.view(T,B,-1,3), pose_pred, pose_gt)
            for k, v in metric_dict.items():
                metric_dict_batch[k].append(v*B)
            
    for k, v in metric_dict_batch.items():
        metric_dict_batch[k] = np.sum(v)/data_size
    print(metric_dict_batch)

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser(description='Eval')
    subparsers =  parser.add_subparsers(help='sub-command help')
    parser_diffusion = subparsers.add_parser('diffusion')
    parser_diffusion.add_argument("--mode", type=str, default='train')
    parser_diffusion.add_argument("--model", type=str, default='Diffusion')
    parser_diffusion.add_argument("--num_joints", type=int, default=21)
    parser_diffusion.add_argument("--num_points", type=int, default=12)

    # transformer
    parser_diffusion.add_argument("--latent_dim", type=int, default=256)
    parser_diffusion.add_argument("--embedding_dim", type=int, default=256)
    parser_diffusion.add_argument("--num_heads", type=int, default=4)
    parser_diffusion.add_argument("--ff_size", type=int, default=256)
    parser_diffusion.add_argument("--activation", type=str, default='gelu')
    parser_diffusion.add_argument("--num_layers", type=int, default=3)
    parser_diffusion.add_argument("--dropout", type=float, default=0)
    parser_diffusion.add_argument("--latent_usage", type=str, default='memory')

    parser_diffusion.add_argument("--lr", type=float, default=3e-4)
    parser_diffusion.add_argument("--l2_norm", type=float, default=0)

    parser_diffusion.add_argument("--weight_past", type=float, default=0.5)
    parser_diffusion.add_argument("--weight_body", type=float, default=2)
    parser_diffusion.add_argument("--weight_obj", type=float, default=1)
    parser_diffusion.add_argument("--weight_obj_rot", type=float, default=1)
    parser_diffusion.add_argument("--weight_obj_nonrot", type=float, default=1)

    parser_diffusion.add_argument("--weight_quat_reg", type=float, default=0.01)
    parser_diffusion.add_argument("--weight_v", type=float, default=1)

    # dataset
    parser_diffusion.add_argument("--past_len", type=int, default=10)
    parser_diffusion.add_argument("--future_len", type=int, default=10)
    parser_diffusion.add_argument("--align_data", default=False, action='store_true')
    parser_diffusion.add_argument("--discard_discrep", default=False, action='store_true')

    # train
    parser_diffusion.add_argument("--batch_size", type=int, default=64)
    parser_diffusion.add_argument("--num_workers", type=int, default=4)
    parser_diffusion.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser_diffusion.add_argument("--gpus", type=int, default=1)
    parser_diffusion.add_argument("--max_epochs", type=int, default=200)

    parser_diffusion.add_argument("--expr_name", type=str, default=datetime.now().strftime("%b-%j-%H:%M"))
    parser_diffusion.add_argument("--render_epoch", type=int, default=5)
    parser_diffusion.add_argument("--debug", type=int, default=0)

    # diffusion
    parser_diffusion.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                        help="Noise schedule type")
    parser_diffusion.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    parser_diffusion.add_argument("--diffusion_steps", type=int, default=1000)
    parser_diffusion.add_argument("--cond_mask_prob", default=0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    parser_diffusion.add_argument("--diverse_samples", type=int, default=10)
    parser_diffusion.add_argument("--resume_checkpoint", type=str, default="./checkpoints/diffusion_v2.ckpt")
    args_diffusion = parser_diffusion.parse_args()
    args_diffusion.smpl_dim = args_diffusion.num_joints*3

    obj_args = dict(
        past_len = 10,
        future_len = 10,
        num_joints=21,
        num_points=12,
        latent_dim = 128,
        embedding_dim = 128,
        activation = 'gelu',
        dropout = 0.1,
        lr = 1e-4,
        weight_smplx_rot = 1,
        weight_smplx_nonrot = 0.1,
        weight_obj_rot = 0.1,
        weight_obj_nonrot = 0.1,
        weight_past = 0.5,
        weight_v = 1
    )

    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results/test"
    os.makedirs(results_folder, exist_ok=True)
    # load test dataset
    train_set, val_set, test_set, unseen_test_set = get_datasets()
    save_dir = Path(os.path.join(results_folder,args_diffusion.expr_name))

    #pin_memory cause warning in pytorch 1.9.0
    test_loader_unseen = DataLoader(unseen_test_set, batch_size=args_diffusion.batch_size, num_workers=1, shuffle=False,
                    drop_last=False, pin_memory=False)
    test_loader_seen = DataLoader(test_set, batch_size=args_diffusion.batch_size, num_workers=1, shuffle=False,
                    drop_last=False, pin_memory=False)  
    print('dataset loaded')

    model = LitInteraction.load_from_checkpoint(args_diffusion.resume_checkpoint, args=args_diffusion).to(device)
    
    model.eval()
    print('start evaluation')
    print('seen test set')
    sample(4, test_loader_seen, len(test_set))
    print('unseen test set')
    sample(4, test_loader_unseen, len(unseen_test_set))