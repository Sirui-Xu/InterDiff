import sys
directory = os.path.dirname(os.path.abspath(__file__))
# setting path
import numpy as np
import pickle
import pathlib
from torch.utils.data import Dataset, random_split
import torch
from scipy.spatial.transform import Rotation as Rot
import pickle
import os
import yaml

with open(directory + "/cfg/HOI.yml", 'r') as stream:
    paths = yaml.safe_load(stream)

MOTION_PATH = paths['MOTION_PATH']

counter=0
def parse_paths():
    """
    parse all sequences
    return a list of filenames and attributes
    """
    files = [] # (path, name, other attributes)
    for p in pathlib.Path(MOTION_PATH).iterdir():
        assert p.is_dir()
        if len(list(p.iterdir()))==0:
            continue
        assert len(list(p.iterdir()))==1, p
        for f in p.iterdir():
            # parse attributes
            filename = f.stem
            object_name = filename.split('_')[1]
            files.append((str(f),filename, object_name))
    # "We collected a dataset of 508 human-object interaction videos"
    print(f'collected sequences: {len(files)}')
    return files

def recover_init_obj(initial_obj, initial_pose):
    """
    :param initial_obj: [N_points, 3]
    :param initial_pose: [7]
    :return: [N_points, 3]
    """
    initial_translation = initial_pose[:3][None] # 1,3
    initial_quat = initial_pose[-4:]# 4
    initial_trans_matrix = Rot.from_quat(initial_quat).inv().as_matrix()# 3,3
    translated_obj = (initial_obj - initial_translation)[:,:,None] # N_points,3,1
    rotated_obj = np.matmul(initial_trans_matrix[None],translated_obj)[:,:,0]#N_points,3,1
    return rotated_obj

def get_consistent_poses(downsampled_poses):
    """
    :param downsampled_poses: [T, 7]
    avoid quaternion flipping problem, that could lead to instable training
    """
    dist_f = lambda arr1,arr2: np.linalg.norm(arr1[-4:] - arr2[-4:], axis=0,ord=2)
    T = downsampled_poses.shape[0]
    downsampled_poses_ret = downsampled_poses.copy()
    for i in range(T-1):
        if dist_f(downsampled_poses_ret[i], downsampled_poses_ret[i+1]) > dist_f(downsampled_poses_ret[i], -downsampled_poses_ret[i+1]):
            downsampled_poses_ret[i+1,-4:] = -downsampled_poses_ret[i+1,-4:]

    return downsampled_poses_ret

def pose_init_to_seq(zero_pose_obj, poses):
    """
    :param zero_pose_obj: [N_points, 3]
    :param poses: [T, 7]
    :return: [T, N_points, 3]
    R*i+t
    """
    
    rots = Rot.from_quat(poses[:,-4:])
    translation_arr = poses[:,None, :3,None] # N_frames,1, 3,1 
    rots_arr = rots.as_matrix()[:,None,:,:] # N_frames,1,3,3

    obj_base = zero_pose_obj[None,:,:,None] # 1,N_points, 3,1

    obj_pred = translation_arr + np.matmul(rots_arr,obj_base)
    return obj_pred[:,:,:,0]


def check_sequences(poseData, objData,discard_discrep):
    # NOTE: make sure *near* zero-pose at start
    objData = objData[::12].copy()
    poseData = poseData[::12].copy()
    # recover zero-pose
    zero_pose_obj = recover_init_obj(objData[0], poseData[0])# N_joints, 3

    # make sure quaternions are valid
    assert (np.linalg.norm(poseData[:,-4:],ord=2,axis=-1)-1).sum()<1e-4, (np.linalg.norm(poseData[:-4:],ord=2,axis=-1)-1).sum()

    # NOTE: check pose-point consistency
    # 545 sequences, 35 not consistent
    if discard_discrep:
        obj_pred = pose_init_to_seq(zero_pose_obj, poseData)
        discrepancy = np.linalg.norm(obj_pred-objData, axis=-1,ord=2).mean()
        if discrepancy>1e-2:
            global counter
            counter = counter + 1
            print(f'error {counter} encoutered')
            return False, zero_pose_obj

    return True, zero_pose_obj

def get_sequences(pathName, discard_discrep=False,unseen=False,filename = None, obj_name=None):
    with open(pathName, 'rb') as f:
        dataList = pickle.load(f)[0]

    startFrame = 0
    endFrame = len(dataList[0])
    skeledonData = dataList[0][startFrame:endFrame:1][:]
    skeledonData = np.array(skeledonData,dtype='float64').reshape((len(dataList[0]), 21, 3))    
    objData = dataList[3][startFrame:endFrame:1][:]
    objData = np.array(objData,dtype='float64').reshape((len(dataList[0]), 12, 3))
    poseData = dataList[2][startFrame:endFrame:1][:]
    poseData = np.array(poseData,dtype='float64').reshape((len(dataList[0]), 7))
    contactData = dataList[1][startFrame:endFrame:1][:]
    contactData = np.array(contactData, dtype='float64').reshape((len(dataList[0]), 1))
    if contactData.sum()<0.5 and unseen:
        print(f'no enough contact, skipping {pathName}')
        return []

    valid, zero_pose_obj = check_sequences(poseData, objData,discard_discrep)
    if not valid:
        print('sequence not valid')
        return []

    # step 1: sliding window
    # "We use a sliding window of 240 frames and a step size of 12
    # frames to extract sequences from a given MoCap video"

    # NOTE: check quaternion continuity
    # not continuous, quat flipping detected, many, not a few
    poseData_ret = get_consistent_poses(poseData)
    # quat_diff = np.linalg.norm(poseData[:-1,-4:] - poseData[1:,-4:], axis=-1,ord=2)

    # quat_diff_ret = np.linalg.norm(poseData_ret[:-1,-4:] - poseData_ret[1:,-4:], axis=-1,ord=2)
    # assert  (quat_diff_ret < 1.5).all(), (quat_diff_ret,quat_diff)
    poseData = poseData_ret

    interval_start = 0
    sequences = []#(skeleton, objData)
    while (interval_start + 240) < endFrame:
        # "each extracted sequence has 20 frames equivalent to a 2-sec motion"
        downsampled_skeledon = skeledonData[interval_start:interval_start +240][::12].copy()
        downsampled_obj = objData[interval_start:interval_start +240][::12].copy()
        downsampled_poses = poseData[interval_start:interval_start +240][::12].copy()
        downsampled_contact = contactData[interval_start:interval_start +240][::12].copy()

        # check contact
        if downsampled_contact.sum()<0.5 and unseen:
            pass
        else:
            sequences.append((downsampled_skeledon, \
            downsampled_obj, downsampled_poses, zero_pose_obj, filename, obj_name)) 

        interval_start = interval_start + 12

    return sequences

def get_datasets(align_data=False,discard_discrep=False):
    
    # step 2:data split
    # we reserved all 471 samples of Chairs 3 and 4 (see Fig. 2) to form a unseen instance test set
    # return: 'train', 'valid', 'test_seen', 'test_unseen'
    ds_save_dir = MOTION_PATH + "/ds_seen.pkl"
    ds_unseen_save_dir = MOTION_PATH + "/ds_test_unseen.pkl"
    if os.path.exists(ds_save_dir):
        print('loading from file')
        with open(ds_save_dir,'rb') as f:
            sequences_seen = pickle.load(f)
        with open(ds_unseen_save_dir,'rb') as f:
            sequences_unseen = pickle.load(f)
    else:
        sample_files = parse_paths()
        print(len(sample_files))
        sequences_seen = []
        sequences_unseen = []
        for i,(f, filename, object_name) in enumerate(sample_files):
            if i%50==0:
                print(f'loaded sequences {i}')
            unseen = (object_name in ["chair3","chair4"])
            
            sample_sequences = get_sequences(f,align_data,discard_discrep,unseen,filename=filename,obj_name=object_name)
            if len(sample_sequences)>1:
                if unseen:
                    sequences_unseen.extend(sample_sequences)
                else:
                    sequences_seen.extend(sample_sequences)
        
        with open(ds_save_dir,'wb') as f:
            pickle.dump(sequences_seen, f)
        with open(ds_unseen_save_dir,'wb') as f:
            pickle.dump(sequences_unseen, f)
    # 0.7,0.2,0.1
    seen_size = len(sequences_seen)
    print(seen_size)
    print(f'size of unseen test dataset: {len(sequences_unseen)}')
    train_set, valid_set, test_set = random_split(SimpleDataset(sequences_seen), [int(0.7*seen_size), \
    int(0.2*seen_size), seen_size - int(0.2*seen_size)- int(0.7*seen_size)], generator=torch.Generator().manual_seed(42))

    return train_set, valid_set, test_set, SimpleDataset(sequences_unseen)

def get_unseen_dataset():
    sample_files = parse_paths()
    sequences_unseen = []

    for i,(f, filename, object_name) in enumerate(sample_files):
        
        if object_name in ["chair3", "chair4"]:
            sample_sequences = get_sequences(f,filename=filename,obj_name=object_name)
            sequences_unseen.extend(sample_sequences)

    # 0.7,0.2,0.1
    return SimpleDataset(sequences_unseen)

class SimpleDataset(Dataset):
    def __init__(self,sequences):
        self.sequences = sequences

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return len(self.sequences)

if __name__=='__main__':
    train_set, val_set, test_set, unseen_test_set = get_datasets()
    print(len(unseen_test_set))