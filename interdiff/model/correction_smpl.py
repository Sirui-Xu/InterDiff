import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from data.utils import marker2bodypart
from model.layers import ST_GCNN_layer

class ObjProjector(nn.Module):
    def __init__(self, args):
        super(ObjProjector, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        self.n_pre = args.dct
        self.st_gcnns_relative=nn.ModuleList()
        self.st_gcnns_relative.append(ST_GCNN_layer(9,32,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(32,16,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(16,32,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

        self.st_gcnns_relative.append(ST_GCNN_layer(32,9,[1,1],1,self.n_pre,
                                               args.num_verts,args.dropout,version=0))

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
        self.st_gcnns_all.append(ST_GCNN_layer(9,32,[1,1],1,self.n_pre,
                                               args.num_verts+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(32,16,[1,1],1,self.n_pre,
                                               args.num_verts+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(16,32,[1,1],1,self.n_pre,
                                               args.num_verts+1,args.dropout,version=2))

        self.st_gcnns_all.append(ST_GCNN_layer(32,9,[1,1],1,self.n_pre,
                                               args.num_verts+1,args.dropout,version=2))

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

    def forward(self, data, initialize=False):
        obj_angles = torch.cat([frame['objfit_params']['angle'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBx3
        obj_angles = matrix_to_rotation_6d(axis_angle_to_matrix(obj_angles))
        obj_trans = torch.cat([frame['objfit_params']['trans'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBx3
        human_verts = torch.cat([frame['markers'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBxPx4
        contact = human_verts[self.args.past_len:, :, :, 6].sum(dim=0) # B P
        final_results = self.sample(obj_angles, obj_trans, human_verts, contact, initialize)
        obj_gt = torch.cat([obj_angles, obj_trans], dim=2)
        return final_results, obj_gt

    def sample(self, obj_angles, obj_trans, human_verts, contact, initialize=False):
        human_verts = human_verts[:, :, :, :3]
        dct_m = self.dct_m.to(obj_angles.device).float()
        idct_m = self.idct_m.to(obj_angles.device).float()

        idx_pad = list(range(self.args.past_len)) + [self.args.past_len - 1] * self.args.future_len

        obj_trans_relative = obj_trans.unsqueeze(2) - human_verts[:, :, :, :3]
        obj_relative = torch.cat([obj_angles.unsqueeze(2).repeat(1, 1, obj_trans_relative.shape[2], 1), obj_trans_relative], dim=3)[idx_pad]
        T, B, P, C = obj_relative.shape
        obj_relative = obj_relative.permute(1, 0, 3, 2).contiguous().view(B, T, C * P)
        obj_relative = torch.matmul(dct_m[:self.n_pre], obj_relative).view(B, -1, C, P).permute(0, 2, 1, 3).contiguous() # B C T P

        x = obj_relative.clone()
        for gcn in (self.st_gcnns_relative):
            x = gcn(x)

        obj_relative = obj_relative + x
        human_trans = human_verts[:, :, :, :3].permute(1, 0, 3, 2).contiguous().view(B, T, -1)
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
        results = torch.matmul(idct_m[:, :self.n_pre], obj).view(B, T, C, P+1).permute(1, 0, 3, 2)[:, :, :, :9]

        if initialize:
            final_results = results.mean(dim=2)
        else:
            final_results = torch.zeros((T, B, 9)).to(results.device)
            final_results[:, contact.sum(dim=1) == 0] = results[:, contact.sum(dim=1) == 0, 0, :]
            contact_happen = contact[contact.sum(dim=1) > 0].float()
            hand_marker = marker2bodypart["left_hand_ids"] + marker2bodypart["right_hand_ids"]
            contact_happen[:, hand_marker] = contact_happen[:, hand_marker] + 0.5
            results_contact_happen = results[:, contact.sum(dim=1) > 0, 1:, :]
            if self.training:
                idx = torch.multinomial(contact_happen, 1).unsqueeze(0).unsqueeze(3).repeat(T, 1, 1, 9) # B, 1
            else:
                idx = torch.argmax(contact_happen, dim=1, keepdim=True).unsqueeze(0).unsqueeze(3).repeat(T, 1, 1, 9) # B, 1
        
            final_results[:, contact.sum(dim=1) > 0] = torch.gather(results_contact_happen, 2, idx).squeeze(2)

        return final_results