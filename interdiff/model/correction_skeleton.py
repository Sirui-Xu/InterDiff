import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_quaternion
from model.layers import ST_GCNN_layer

class ObjProjector(nn.Module):
    def __init__(self, args):
        super(ObjProjector, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        self.n_pre = 20
        self.st_gcnns_relative=nn.ModuleList()
        self.st_gcnns_relative.append(ST_GCNN_layer(9,32,[1,1],1,self.n_pre,
                                               args.num_joints,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(32,16,[1,1],1,self.n_pre,
                                               args.num_joints,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(16,32,[1,1],1,self.n_pre,
                                               args.num_joints,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(32,9,[1,1],1,self.n_pre,
                                               args.num_joints,args.dropout,version=0))

        self.st_gcnns=nn.ModuleList()
        self.st_gcnns.append(ST_GCNN_layer(9,32,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns.append(ST_GCNN_layer(32,16,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns.append(ST_GCNN_layer(16,32,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns.append(ST_GCNN_layer(32,9,[1,1],1,self.n_pre,
                                               1,args.dropout,version=0))

        self.st_gcnns_all=nn.ModuleList()
        self.st_gcnns_all.append(ST_GCNN_layer(9,64,[1,1],1,self.n_pre,
                                               args.num_joints+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(64,32,[1,1],1,self.n_pre,
                                               args.num_joints+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(32,64,[1,1],1,self.n_pre,
                                               args.num_joints+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(64,9,[1,1],1,self.n_pre,
                                               args.num_joints+1,args.dropout,version=2))

        self.dct_m, self.idct_m = self.get_dct_matrix(args.past_len + args.future_len)

    def get_dct_matrix(self, N, is_torch=True):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        if is_torch:
            dct_m = torch.from_numpy(dct_m)
            idct_m = torch.from_numpy(idct_m)
        return dct_m, idct_m     

    def forward(self, obj_angles, obj_trans, human_points):
        # NOTE: align data format
        # obj_angles: T,B,4
        # obj_trans: T,B,3
        # human_points: T,B,N_joints,3
        obj_angles_gt = obj_angles.clone()
        quat_correct = torch.cat([obj_angles[:,:,-1,None], obj_angles[:,:,-4:-1]],dim=2)
        obj_angles = matrix_to_rotation_6d(quaternion_to_matrix(quat_correct))
        assert not obj_angles.isnan().any()
        obj_trans_gt = obj_trans.clone()        

        obj_angles_p, obj_trans_p = self.sample(obj_angles, obj_trans, human_points)
        return obj_angles_p, obj_trans_p, obj_angles_gt, obj_trans_gt 



    def sample(self, obj_angles, obj_trans, human_points):
        # TODO: align data format
        # obj_angles: T,B,4
        # obj_trans: T,B,3
        # human_points: T,B,N_joints,3
        quat_correct = torch.cat([obj_angles[:,:,-1,None], obj_angles[:,:,-4:-1]],dim=2)
        obj_angles = matrix_to_rotation_6d(quaternion_to_matrix(quat_correct))

        dct_m = self.dct_m.to(obj_angles.device).float()
        idct_m = self.idct_m.to(obj_angles.device).float()

        idx_pad = list(range(self.args.past_len)) + [self.args.past_len - 1] * self.args.future_len

        obj_trans_relative = obj_trans.unsqueeze(2) - human_points
        obj_relative = torch.cat([obj_angles.unsqueeze(2).repeat(1, 1, obj_trans_relative.shape[2], 1), obj_trans_relative], dim=3)[idx_pad]
        T, B, P, C = obj_relative.shape
        obj_relative = obj_relative.permute(1, 0, 3, 2).contiguous().view(B, T, C * P)
        obj_relative = torch.matmul(dct_m[:self.n_pre], obj_relative).view(B, -1, C, P).permute(0, 2, 1, 3).contiguous() # B C T P

        x = obj_relative.clone()
        for gcn in (self.st_gcnns_relative):
            x = gcn(x)

        obj_relative = obj_relative + x
        human_trans = human_points.permute(1, 0, 3, 2).contiguous().view(B, T, -1)
        human_trans = torch.matmul(dct_m[:self.n_pre], human_trans).view(B, -1, 3, P).permute(0, 2, 1, 3).contiguous() # B C T P
        obj_multi = torch.cat([obj_relative[:, :6, :, :], obj_relative[:, 6:9, :, :] + human_trans], dim=1)

        obj_gt = torch.cat([obj_angles, obj_trans], dim=2)
        obj = obj_gt[idx_pad].unsqueeze(2)
        obj = obj.permute(1, 0, 3, 2).contiguous().view(B, T, C * 1)
        obj = torch.matmul(dct_m[:self.n_pre], obj).view(B, -1, C, 1).permute(0, 2, 1, 3).contiguous() # B C T P

        x = obj.clone()
        for gcn in (self.st_gcnns):
            x = gcn(x)

        obj = obj + x
        obj = torch.cat([obj, obj_multi], dim=3)

        x = obj.clone()
        for gcn in (self.st_gcnns_all):
            x = gcn(x)

        obj = obj + x
        obj = obj.permute(0, 2, 1, 3).contiguous().view(B, -1, C * (P+1))
        results = torch.matmul(idct_m[:, :self.n_pre], obj).view(B, T, C, P+1).permute(1, 0, 3, 2)[:, :, 0, :9]

        obj_angles_p = matrix_to_quaternion(rotation_6d_to_matrix(results[:,:,:6]))
        obj_angles_p = torch.cat([obj_angles_p[:,:,1:4], obj_angles_p[:,:,0,None]],dim=2)
        obj_trans_p = results[:,:,6:9]
        return obj_angles_p, obj_trans_p