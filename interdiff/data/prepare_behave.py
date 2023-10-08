"""
Adapted from: 
    Code to generate contact labels from SMPL and object registrations
    Author: Xianghui Xie
    Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
import numpy as np
sys.path.append(os.getcwd())
directory = os.path.dirname(os.path.abspath(__file__))
import trimesh
import igl
from os.path import isfile
from psbody.mesh import Mesh
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
import yaml
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json

with open(directory + "/cfg/BEHAVE.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
SPLIT_PATH, OBJECT_PATH, MOTION_PATH, MODEL_PATH = paths['SPLIT_PATH'], paths['OBJECT_TEMPLATE'], paths['MOTION_PATH'], paths['MODEL_PATH']

class ContactLabelGenerator(object):
    "class to generate contact labels"
    def __init__(self):
        pass
        

    def get_contact_labels(self, smpl, object_points, thres=0.02):
        """
        sample point on the object surface and compute contact labels for each point
        :param smpl: trimesh object
        :param obj: trimesh object
        :param num_samples: number of samples on object surface
        :param thres: threshold to determine whether a point is in contact with the human
        :return:
        for each point: a binary label (contact or not) and the closest SMPL vertex
        """
        dist, faces, vertices, normals = igl.signed_distance(object_points, smpl.vertices, smpl.faces, return_normals=True)

        # smpl_all = np.concatenate([vertices, normals], axis=1)
        contact_object_label = np.where(dist<thres)[0]
        # print(dist<thres, object_points[np.where(dist<thres)[0]].shape)
        # smpl.vertices - object_points[np.where(dist<thres)[0]]
        vertices_1 = np.expand_dims(object_points[contact_object_label], axis=0)
        vertices_2 = np.expand_dims(smpl.vertices, axis=1)

        contact_human_label = np.where((np.linalg.norm((vertices_1 - vertices_2), axis=2) < thres).any(axis=1))[0]
        return contact_object_label, contact_human_label

    def to_trimesh(self, mesh):
        tri = trimesh.Trimesh(mesh.v, mesh.f, process=False)
        return tri


def main(args, name):
    outfile = os.path.join(MOTION_PATH, name, 'contact.npz')
    if isfile(outfile):
        print(outfile, 'done, skipped')
        return
    with np.load(os.path.join(MOTION_PATH, name, 'object_fit_all.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, frame_times = f['angles'], f['trans'], f['frame_times']
    with np.load(os.path.join(MOTION_PATH, name, 'smpl_fit_all.npz'), allow_pickle=True) as f:
        poses, betas, trans = f['poses'], f['betas'], f['trans']
    
    info_file = os.path.join(MOTION_PATH, name, 'info.json')

    info = json.load(open(info_file))
    gender = info['gender']
    obj_name = info['cat']
    batch_end = len(frame_times)
    generator = ContactLabelGenerator()
    mesh_obj = Mesh()
    mesh_obj.load_from_obj(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"))
    smpl_male = SMPL_Layer(center_idx=0, gender='male', num_betas=10,
                                model_root=str(MODEL_PATH), hands=True)
    smpl_female = SMPL_Layer(center_idx=0, gender='female', num_betas=10,
                            model_root=str(MODEL_PATH), hands=True)
    smpl = {'male': smpl_male, 'female': smpl_female}[gender]
    verts, jtr, _, _ = smpl(torch.tensor(poses), th_betas=torch.tensor(betas), th_trans=torch.tensor(trans))
    verts = np.float32(verts)
    faces = smpl.th_faces.numpy()

    obj_verts = mesh_obj.v
    obj_faces = mesh_obj.f
    center = np.mean(obj_verts, 0)
    obj_verts = obj_verts - center
    obj = trimesh.Trimesh(obj_verts, obj_faces, process=False)
    object_points, object_faces = obj.sample(args.num_samples, return_index=True)
    object_normals = obj.face_normals[object_faces]
    object_all = np.concatenate([object_points, object_normals], axis=1)
    
    contact_dict = {
        "object_points": object_all,
        'object_contact_vertex_label':[],
        'human_contact_vertex_label':[],
        'foot_contact_joint_label':[],
    }
    for idx in tqdm(range(batch_end)): 
        smpl = trimesh.Trimesh(verts[idx], faces, process=False)
        joints = jtr[idx]
        # foot_contact = joints[10] if joints[10, 1] > joints[11, 1] else joints[11]
        foot_contact_label = 10 if joints[10, 1] > joints[11, 1] else 11
        obj_v = object_points.copy()
        a, t = obj_angles[idx], obj_trans[idx]
        rot = Rotation.from_rotvec(a).as_matrix()
        obj_v = np.matmul(obj_v, rot.T) + t
        # transform canonical mesh to fitting
        contact_object_label, contact_human_label = generator.get_contact_labels(
            smpl, obj_v
        )
        contact_dict["object_contact_vertex_label"].append(contact_object_label)
        contact_dict["human_contact_vertex_label"].append(contact_human_label)
        contact_dict["foot_contact_joint_label"].append(foot_contact_label)

    np.savez(outfile, contact_dict)
        # print(contacts, contact_vertices, contacts.sum(), contact_vertices.sum())
    print('all done for ', name)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-fs', '--start', type=int, default=0, help='index of the start frame')
    parser.add_argument('-fe', '--end', type=int, default=None)
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    parser.add_argument('-redo', default=True, action='store_false')

    args = parser.parse_args()
    print(os.listdir(MOTION_PATH))
    for data in (os.listdir(MOTION_PATH)):
        main(args, data)


