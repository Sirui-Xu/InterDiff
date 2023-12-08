import os
import numpy as np
import torch
from data.dataset_smpl import Dataset, MODEL_PATH, OBJECT_PATH
from data.tools import vertex_normals
from data.utils import SIMPLIFIED_MESH
from tools import point2point_signed
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from torch.autograd import Variable
import torch.optim as optim
import copy
from render.mesh_viz import visualize_body_obj
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def optimize(index, data, visualize=False):
    body_pose = torch.stack([torch.from_numpy(frame['smplfit_params']['pose'][:66]) for frame in data['frames']], dim=0).float().to(device) # TxDb
    body_trans = torch.stack([torch.from_numpy(frame['smplfit_params']['trans']) for frame in data['frames']], dim=0).float().to(device) # Tx3
    obj_angles = torch.stack([torch.from_numpy(frame['objfit_params']['angle']) for frame in data['frames']], dim=0).float().to(device) # Tx3
    obj_trans = torch.stack([torch.from_numpy(frame['objfit_params']['trans']) for frame in data['frames']], dim=0).float().to(device) # Tx3
    obj_points = torch.from_numpy(data['obj_points'][:, :3]).float().to(device)
    hand_pose = torch.stack([torch.from_numpy(frame['smplfit_params']['pose'][66:]) for frame in data['frames']], dim=0).float().to(device)
    
    T, _ = body_pose.shape
    obj_rot = axis_angle_to_matrix(obj_angles)
    glo_rot = axis_angle_to_matrix(body_pose[:, :3].view(T, -1, 3))
    body_rot = axis_angle_to_matrix(body_pose[:, 3:].view(T, -1, 3))
    hand_rot = axis_angle_to_matrix(hand_pose.view(T, -1, 3))
    
    
    smpl_male = SMPL_Layer(center_idx=0, gender='male', num_betas=10,
                                model_root=str(MODEL_PATH), hands=True).to(device)
    smpl_female = SMPL_Layer(center_idx=0, gender='female', num_betas=10,
                            model_root=str(MODEL_PATH), hands=True).to(device)
    body_model = {'male': smpl_male, 'female': smpl_female}
    smpl = body_model[data['gender']]
    beta = torch.stack([torch.from_numpy(frame['smplfit_params']['betas']) for frame in data['frames']], dim=0).to(device)

    verts_gt, jtr_gt, _, _ = smpl(torch.cat([body_pose, hand_pose], dim=1).to(device), 
                                th_betas=beta, 
                                th_trans=body_trans.to(device))
    left_foot = jtr_gt[:, 10]
    right_foot = jtr_gt[:, 11]
    delta_left = torch.norm(left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]], dim=1) + 1e-6
    delta_right = torch.norm(right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]], dim=1) + 1e-6
    left_static = (delta_left < 0.008)
    right_static = (delta_right < 0.008)
    
    def calc_loss(body_rec, transl_rec, glo_rot_rec, obj_transl_rec, obj_rot_rec, hand_pose_rec, ratio): 
        with torch.enable_grad():       
            pose = matrix_to_axis_angle(torch.cat([glo_rot_rec, body_rec, hand_pose_rec], dim=1)).view(T, -1)
            verts, jtr, _, _ = smpl(pose, 
                                    th_betas=beta, 
                                    th_trans=transl_rec)

            obj_transl = obj_transl_rec
            obj_points_pred = torch.matmul(obj_points.unsqueeze(0), obj_rot_rec.permute(0, 2, 1)) + obj_transl.unsqueeze(1)
            normals = vertex_normals(verts, smpl.th_faces.unsqueeze(0).repeat(verts.shape[0], 1, 1))
            o2h_signed, h2o_signed, o2h_idx, h2o_idx, o2h, h2o = point2point_signed(verts.view(T, -1, 3), obj_points_pred.view(T, -1, 3), x_normals=normals, return_vector=True)

            w = torch.zeros([T, o2h_signed.size(1)]).to(o2h_signed.device)
            w_dist = (o2h_signed < 0.01) * (o2h_signed > 0)
            w_dist_neg = o2h_signed < 0
            w[w_dist] = 0 # small weight for far away vertices
            w[w_dist_neg] = 20 * ratio if ratio < 1 else 20 # large weight for penetration

            w_verts = torch.zeros([T, verts.shape[1]]).to(o2h_signed.device) + 1e-2
            distance = torch.norm(verts.unsqueeze(1) - obj_points_pred.view(T, -1, 3).unsqueeze(2), dim=3)
            contact_human_label = (distance < 0.5).any(dim=1)
            w_verts[contact_human_label] = 0
            
            loss_verts_reg = torch.einsum('ij,ij->ij', torch.abs(verts - verts_gt).sum(2), w_verts).sum(dim=1).mean()
            loss_dist_o = torch.einsum('ij,ij->ij', torch.abs(o2h_signed), w).sum(dim=1).mean() # 
                        
            left_foot = jtr[:, 10]
            right_foot = jtr[:, 11]
            if left_static.any():
                loss_left = torch.mean((((left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]])[left_static]) ** 2))
            else:
                loss_left = 0
            if right_static.any():
                loss_right = torch.mean((((right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]])[right_static]) ** 2))
            else:
                loss_right = 0

            loss_obj_transl_reg = 0.1 * torch.mean((obj_transl - obj_trans).abs())
            loss_obj_rot_reg = 0.1 * torch.mean((obj_rot_rec - obj_rot).abs())
            loss_transl_reg = 0.1 * torch.mean((transl_rec - body_trans).abs())
            loss_glo_rot_reg = 0.1 * torch.mean((glo_rot_rec - glo_rot).abs())
            loss_body_reg = 0.005 * torch.mean(((body_rec - body_rot).abs()).sum(dim=2).sum(dim=1))
            # smoothing
            loss_transl_v_reg = 10 * torch.mean(((transl_rec[1:-1] - transl_rec[:-2]) - (transl_rec[2:] - transl_rec[1:-1])) ** 2) + \
                                10 * torch.mean(((transl_rec[1:] - transl_rec[:-1])) ** 2)
            loss_glo_rot_v_reg = 5 * torch.mean(((glo_rot_rec[1:-1] - glo_rot_rec[:-2]) - (glo_rot_rec[2:] - glo_rot_rec[1:-1])) ** 2) + \
                                5 * torch.mean(((glo_rot_rec[1:] - glo_rot_rec[:-1])) ** 2)
            loss_hand_pose_v_reg = 50 * torch.mean(((hand_pose_rec[1:-1] - hand_pose_rec[:-2]) - (hand_pose_rec[2:] - hand_pose_rec[1:-1])) ** 2) + \
                                    50 * torch.mean((hand_pose_rec[1:] - hand_pose_rec[:-1]) ** 2)
            loss_obj_v_reg = 1000 * torch.mean(((obj_transl_rec[1:-1] - obj_transl_rec[:-2]) - (obj_transl_rec[2:] - obj_transl_rec[1:-1])) ** 2) + \
                             100 * torch.mean(((obj_transl_rec[1:] - obj_transl_rec[:-1])) ** 2) +\
                             1000 * torch.mean(((obj_rot_rec[1:-1] - obj_rot_rec[:-2]) - (obj_rot_rec[2:] - obj_rot_rec[1:-1])) ** 2) + \
                             100 * torch.mean(((obj_rot_rec[1:] - obj_rot_rec[:-1])) ** 2)
            loss_body_v_reg = 1000 * torch.mean((((body_rec[1:-1] - body_rec[:-2]) - (body_rec[2:] - body_rec[1:-1])) ** 2).sum(dim=2).sum(dim=1)) + 100 * torch.mean(((body_rec[1:] - body_rec[:-1]) ** 2).sum(dim=2).sum(dim=1)) + 1000 * (loss_left + loss_right)

            loss_v_reg = 1 * (loss_hand_pose_v_reg + loss_obj_v_reg + loss_body_v_reg + loss_transl_v_reg + loss_glo_rot_v_reg)
            loss = (
                    loss_dist_o
                    + (loss_obj_transl_reg + loss_obj_rot_reg + loss_body_reg + loss_transl_reg + loss_glo_rot_reg + loss_verts_reg) +
                    loss_v_reg
                    )
            loss_dict = {}
            loss_dict['total'] = loss.detach().cpu().numpy()

            loss_dict['collision'] = loss_dist_o.detach().cpu().numpy()
            loss_dict['reg'] = (loss_obj_transl_reg + loss_obj_rot_reg + loss_body_reg + loss_transl_reg + loss_glo_rot_reg + loss_verts_reg).detach().cpu().numpy()
            loss_dict['reg_v'] = loss_v_reg.detach().cpu().numpy()
        return loss, loss_dict

    best_eval_grasp = 1e7
    tmp_smplhparams = {}
    tmp_objparams = {}
    obj_transl_rec = Variable(copy.deepcopy(obj_trans).to(device), requires_grad=True)
    obj_rot_rec = Variable(copy.deepcopy(obj_rot).to(device),
                                requires_grad=True)  # 6d

    transl_rec = Variable(copy.deepcopy(body_trans).to(device), requires_grad=True)
    glo_rot_rec = Variable(copy.deepcopy(glo_rot).to(device),
                                requires_grad=True)  # 6d
    body_rec = Variable(copy.deepcopy(body_rot).to(device),
                                requires_grad=True)

    hand_pose_rec = Variable(copy.deepcopy(hand_rot).to(device), requires_grad=True)
    optimizer = optim.Adam([body_rec, transl_rec, glo_rot_rec, obj_transl_rec, obj_rot_rec, hand_pose_rec],
                                lr=0.001)

    for ii in range(200):
        optimizer.zero_grad()
        loss, loss_dict = calc_loss(body_rec, transl_rec, glo_rot_rec, obj_transl_rec, obj_rot_rec, hand_pose_rec, ii / 350)
        losses_str = ' '.join(['{}: {:.4f} | '.format(x, loss_dict[x]) for x in loss_dict.keys()])
        print(losses_str)
        loss.backward(retain_graph=False)
        optimizer.step()
        eval_grasp = loss

        if ii > 150 and eval_grasp < best_eval_grasp:  # and contact_num>=5:
            best_eval_grasp = eval_grasp
            tmp_smplhparams = {}
            tmp_objparams = {}
            tmp_objparams['obj_transl'] = copy.deepcopy(obj_transl_rec.detach()).cpu().numpy()
            tmp_objparams['obj_rot'] = matrix_to_axis_angle(copy.deepcopy(obj_rot_rec.detach())).cpu().numpy()
            
            tmp_smplhparams['transl'] = copy.deepcopy(transl_rec.detach())
            tmp_smplhparams['body_pose'] = matrix_to_axis_angle(copy.deepcopy(body_rec.detach())).view(T, -1)
            tmp_smplhparams['hand_pose'] = matrix_to_axis_angle(copy.deepcopy(hand_pose_rec.detach())).view(T, -1)
            tmp_smplhparams['glo_rot'] = matrix_to_axis_angle(copy.deepcopy(glo_rot_rec.detach())).view(T, -1)

            verts, jtr, _, _ = smpl(torch.cat([tmp_smplhparams['glo_rot'], tmp_smplhparams['body_pose'], tmp_smplhparams['hand_pose']], dim=1), 
                                    th_betas=beta, 
                                    th_trans=tmp_smplhparams['transl'])
            
            tmp_smplhparams['transl'] = tmp_smplhparams['transl'].cpu().numpy()
            tmp_smplhparams['body_pose'] = tmp_smplhparams['body_pose'].cpu().numpy()
            tmp_smplhparams['hand_pose'] = tmp_smplhparams['hand_pose'].cpu().numpy()
            tmp_smplhparams['glo_rot'] = tmp_smplhparams['glo_rot'].cpu().numpy()

    body_pose_new = np.concatenate([tmp_smplhparams['glo_rot'], tmp_smplhparams['body_pose'], tmp_smplhparams['hand_pose']], axis=1)
    for i in range(T):
        data['frames'][i]['smplfit_params']['pose'] = body_pose_new[i]
        data['frames'][i]['smplfit_params']['trans'] = tmp_smplhparams['transl'][i]
        data['frames'][i]['objfit_params']['angle'] = tmp_objparams['obj_rot'][i]
        data['frames'][i]['objfit_params']['trans'] = tmp_objparams['obj_transl'][i]
    
    if visualize:
        verts = verts.detach().cpu().numpy()
        verts_gt = verts_gt.detach().cpu().numpy()
        obj_verts = []
        obj_verts_gt = []
        # visualize
        export_file = "./optimization/render/"
        os.makedirs(export_file, exist_ok=True)
        # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
        rend_video_path = os.path.join(export_file, '{}_s{}_{}.gif'.format(index, data['start_frame'], T))
        rend_video_path_optim = os.path.join(export_file, '{}_s{}_{}_optim.gif'.format(index, data['start_frame'], T))
        T, _ = body_pose.shape
        for t in range(T):
            mesh_obj = Mesh()
            mesh_obj.load_from_file(os.path.join(OBJECT_PATH, SIMPLIFIED_MESH[data['obj_name']]))
            mesh_obj_v = mesh_obj.v.copy()
            # center the meshes
            center = np.mean(mesh_obj_v, 0)
            mesh_obj_v = mesh_obj_v - center
            angle, trans = tmp_objparams['obj_rot'][t], tmp_objparams['obj_transl'][t]
            rot = Rotation.from_rotvec(angle).as_matrix()
            # transform canonical mesh to fitting
            mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
            obj_verts.append(mesh_obj_v)

            mesh_obj_v_gt = mesh_obj.v.copy()
            # center the meshes
            center = np.mean(mesh_obj_v_gt, 0)
            mesh_obj_v_gt = mesh_obj_v_gt - center
            angle, trans = obj_angles.cpu().numpy()[t], obj_trans.cpu().numpy()[t]
            rot = Rotation.from_rotvec(angle).as_matrix()
            # transform canonical mesh to fitting
            mesh_obj_v_gt = np.matmul(mesh_obj_v_gt, rot.T) + trans
            obj_verts_gt.append(mesh_obj_v_gt)

        m1 = visualize_body_obj(verts, smpl.th_faces.cpu().numpy(), np.array(obj_verts), mesh_obj.f, past_len=10, save_path=rend_video_path_optim, sample_rate=1)
        m2 = visualize_body_obj(verts_gt, smpl.th_faces.cpu().numpy(), np.array(obj_verts_gt), mesh_obj.f, past_len=10, save_path=rend_video_path, sample_rate=1)

    return data
    

if __name__ == '__main__':
    test_dataset = Dataset(mode = 'test', past_len=10, future_len=10)
    for i, data in enumerate(test_dataset):
        optimize(i, data, True)