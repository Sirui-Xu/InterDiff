import json
import os
import os.path
import sys
directory = os.path.dirname(os.path.abspath(__file__))
# setting path
sys.path.append(os.path.dirname(directory))
import numpy as np

import torch

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import yaml
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from data.tools import vertex_normals
from data.utils import markerset_ssm67_smplh

with open(directory + "/cfg/BEHAVE.yml", 'r') as stream:
    paths = yaml.safe_load(stream)

SPLIT_PATH, MOTION_PATH, OBJECT_PATH, MODEL_PATH = paths['SPLIT_PATH'], paths['MOTION_PATH'], paths['OBJECT_TEMPLATE'], paths['MODEL_PATH']
print(paths)
class Dataset(Dataset):
    def __init__(self, mode='train', past_len=10, future_len=25, sample_rate=1):
        data_name = os.listdir(MOTION_PATH)
        if mode == 'train':
            data_name = list(filter(lambda x: x[:6] != "Date03", data_name))
        elif mode == 'test':
            data_name = list(filter(lambda x: x[:6] == "Date03", data_name))
        else:
            raise Exception('mode must be train or test.')
        self.past_len = past_len
        self.future_len = future_len
        smpl_male = SMPL_Layer(center_idx=0, gender='male', num_betas=10,
                                model_root=str(MODEL_PATH), hands=True)
        smpl_female = SMPL_Layer(center_idx=0, gender='female', num_betas=10,
                                model_root=str(MODEL_PATH), hands=True)
        self.smpl = {'male': smpl_male, 'female': smpl_female}
        self.data = []
        self.idx2frame = [] # (seq_id, sub_seq_id, bias)
        for k, name in tqdm(enumerate(data_name)):
            with np.load(os.path.join(MOTION_PATH, name, 'object_fit_all.npz'), allow_pickle=True) as f:
                obj_angles, obj_trans, frame_times = f['angles'], f['trans'], f['frame_times']
            with np.load(os.path.join(MOTION_PATH, name, 'smpl_fit_all.npz'), allow_pickle=True) as f:
                poses, betas, trans = f['poses'], f['betas'], f['trans']
            with np.load(os.path.join(MOTION_PATH, name, 'contact.npz'), allow_pickle=True) as f:
                d = f['arr_0'].item()
                object_points, object_contact_vertex_label, human_contact_vertex_label, foot_contact_joint_label = d['object_points'], d['object_contact_vertex_label'], d['human_contact_vertex_label'], d['foot_contact_joint_label']

            frame_times = frame_times.shape[0]
            info_file = os.path.join(MOTION_PATH, name, 'info.json')
            info = json.load(open(info_file))
            gender = info['gender']
            obj_name = info['cat']
            verts, jtr, _, _ = self.smpl[gender](torch.tensor(poses), th_betas=torch.tensor(betas), th_trans=torch.tensor(trans))
            normal_file = os.path.join(MOTION_PATH, name, 'human_normal.npz')
            if os.path.isfile(normal_file):
                with np.load(normal_file, allow_pickle=True) as f:
                    d = f['arr_0'].item()
                    normals = d['normals']
            else:
                normals = vertex_normals(verts, self.smpl[gender].th_faces.unsqueeze(0).repeat(verts.shape[0], 1, 1)).numpy()
                np.savez(normal_file, {"normals":normals})
            normals = torch.from_numpy(normals)
            verts = torch.cat([verts, normals], dim=2)
            pelvis = np.float32(jtr[:, 0])
            left_foot = np.float32(jtr[:, 10])
            right_foot = np.float32(jtr[:, 11])
            records = {
                'gender': gender,
                'obj_name': obj_name,
                'obj_angles': obj_angles,
                'obj_trans': obj_trans,
                'poses': poses,
                'betas': betas,
                'trans': trans,
                'pelvis': pelvis,
                'left_foot': left_foot,
                'right_foot': right_foot,
                'seq_name': name,
                'obj_points': object_points,
                'obj_contact_label': object_contact_vertex_label,
                'human_verts': np.float32(verts),
                'contact_label': human_contact_vertex_label,
                'ground_joint_label': foot_contact_joint_label,
            }
            self.data.append(records)
            fragment = (past_len + future_len) * sample_rate
            for i in range(frame_times // fragment):
                if mode == "test":
                    self.idx2frame.append((k, i * fragment, 1))
                elif i == frame_times // fragment - 1:
                    self.idx2frame.append((k, i * fragment, frame_times + 1 - (frame_times // fragment) * fragment))
                else:
                    self.idx2frame.append((k, i * fragment, fragment))
        self.num_verts = verts.shape[1]
        self.num_markers = len(markerset_ssm67_smplh)
        self.num_obj_points = records['obj_points'].shape[0]
        self.smpl_dim = records['poses'][0].shape[0]
        self.sample_rate = sample_rate
        print("====> The number of clips for " + mode + " data: " + str(len(self.idx2frame)) + " <====")

    def __getitem__(self, idx):
        index, frame_idx, bias = self.idx2frame[idx]
        data = self.data[index]
        start_frame = np.random.choice(bias) + frame_idx
        end_frame = start_frame + (self.past_len + self.future_len) * self.sample_rate
        centroid = None
        rotation = None
        rotation_v = None
        frames = []
        for i in range(start_frame, end_frame, self.sample_rate):
            smplfit_params = {'pose': data['poses'][i].copy(), 'trans': data['trans'][i].copy(), 'betas': data['betas'][i].copy()}
            objfit_params = {'angle': data['obj_angles'][i].copy(), 'trans': data['obj_trans'][i].copy()}
            pelvis = data['pelvis'][i].copy()
            # NOTE: Canonicalize the first human pose
            if i == start_frame:
                centroid = pelvis
                global_orient = Rotation.from_rotvec(smplfit_params['pose'][:3]).as_matrix()
                rotation_v = np.eye(3).astype(np.float32)
                cos, sin = global_orient[0, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2), global_orient[2, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2)
                rotation_v[[0, 2, 0, 2], [0, 2, 2, 0]] = np.array([cos, cos, -sin, sin])
                rotation = np.linalg.inv(rotation_v).astype(np.float32)
            
            smplfit_params['trans'] = smplfit_params['trans'] - centroid
            pelvis = pelvis - centroid
            pelvis_original = pelvis - smplfit_params['trans'] # pelvis position in original smpl coords system
            smplfit_params['trans'] = np.dot(smplfit_params['trans'] + pelvis_original, rotation.T) - pelvis_original
            pelvis = np.dot(pelvis, rotation.T)

            # human vertex in the canonical system
            human_verts_tran = data['human_verts'][i].copy()[:, :3] - centroid
            human_verts_tran = np.dot(human_verts_tran, rotation.T)

            # human vertex normal in the canonical system
            human_verts_normal = np.dot(data['human_verts'][i].copy()[:, 3:], rotation.T)
            human_verts = np.concatenate([human_verts_tran, human_verts_normal], axis=1)

            # smpl pose parameter in the canonical system
            r_ori = Rotation.from_rotvec(smplfit_params['pose'][:3])
            r_new = Rotation.from_matrix(rotation) * r_ori
            smplfit_params['pose'][:3] = r_new.as_rotvec()

            # object in the canonical system
            objfit_params['trans'] = objfit_params['trans'] - centroid
            objfit_params['trans'] = np.dot(objfit_params['trans'], rotation.T)

            r_ori = Rotation.from_rotvec(objfit_params['angle'])
            r_new = Rotation.from_matrix(rotation) * r_ori
            objfit_params['angle'] = r_new.as_rotvec()
            
            # object pointcloud sin the canonical system
            obj_points = data['obj_points'].copy()

            rot = r_new.as_matrix()

            obj_points[:, :3] = np.matmul(obj_points[:, :3], rot.T) + objfit_params['trans']
            obj_points[:, 3:6] = np.matmul(obj_points[:, 3:6], rot.T)

            obj_contact_label = data['obj_contact_label'][i]
            label = np.zeros([obj_points.shape[0], 1])
            label[obj_contact_label, 0] = 1
            obj_points = np.concatenate([obj_points, label], axis=1)

            contact_label = np.zeros([self.num_verts, 1])
            contact_label[data['contact_label'][i], 0] = 1
            human_verts = np.concatenate([human_verts, contact_label], axis=1)

            # The label indicating if the foot is contacting ground
            ground_joint_label = np.zeros([2])
            if i > 0:
                delta_left = np.linalg.norm(data['left_foot'][i] - data['left_foot'][i-1])
                delta_right = np.linalg.norm(data['right_foot'][i] - data['right_foot'][i-1])
                ground_joint_label[0] = int(delta_left < 0.01)
                ground_joint_label[1] = int(delta_right < 0.01)
            else:
                ground_joint_label[data['ground_joint_label'][i] - 10] = 1

            record = {
                'smplfit_params': smplfit_params,
                'objfit_params': objfit_params,
                'pelvis': pelvis,
                'obj_points': obj_points,
                'contact_label': contact_label,
                'ground_joint_label': ground_joint_label,
                'human_verts': human_verts,
                'markers': human_verts[markerset_ssm67_smplh, :]
            }
            frames.append(record)

        records = {
            'gender': data['gender'],
            'centroid': centroid,
            'rotation': rotation,
            'rotation_v': rotation_v,
            'frames': frames,
            'obj_name': data['obj_name'],
            'seq_name': data['seq_name'],
            'start_frame': start_frame,
            'obj_points': data['obj_points'],
        }
        return records

    def __len__(self):
        return len(self.idx2frame)