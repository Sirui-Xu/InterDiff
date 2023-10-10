import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from model.layers import PointNet2Encoder, PositionalEncoding, TimestepEmbedder, TransformerEncoder, TransformerDecoder
from model.sublayers import TransformerDecoderLayerQaN, TransformerEncoderLayerQaN

class MDM(nn.Module):
    def __init__(self, args):
        super(MDM, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        self.bodyEmbedding = nn.Linear(args.smpl_dim+3, num_channels)
        self.pcEmbedding = PointNet2Encoder(c_in=1, c_out=num_channels, num_keypoints=1) if args.use_pointnet2 else nn.Linear(6, num_channels)
        self.objEmbedding = nn.Linear(9, num_channels)
        self.PositionalEmbedding = PositionalEncoding(d_model=num_channels, dropout=args.dropout)
        self.embedTimeStep = TimestepEmbedder(num_channels, self.PositionalEmbedding)
        self.objPooling = torch.nn.MaxPool1d(1)
        from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
        seqTransEncoderLayer1 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer2 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer3 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer4 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer5 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer6 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer7 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer8 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer = nn.ModuleList([seqTransEncoderLayer1, seqTransEncoderLayer2, seqTransEncoderLayer3, seqTransEncoderLayer4,
                                              seqTransEncoderLayer5, seqTransEncoderLayer6, seqTransEncoderLayer7, seqTransEncoderLayer8])
        self.encoder = TransformerEncoder(seqTransEncoderLayer)

        if self.args.latent_usage == 'memory':
            seqTransDecoderLayer1 = TransformerDecoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer2 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer3 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer4 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer5 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer6 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer7 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer8 = TransformerDecoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer = nn.ModuleList([seqTransDecoderLayer1, seqTransDecoderLayer2, seqTransDecoderLayer3, seqTransDecoderLayer4,
                                                  seqTransDecoderLayer5, seqTransDecoderLayer6, seqTransDecoderLayer7, seqTransDecoderLayer8])
            self.decoder = TransformerDecoder(seqTransDecoderLayer)
        else:
            seqTransDecoderLayer1 = TransformerEncoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer2 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer3 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer4 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer5 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer6 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer7 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer8 = TransformerEncoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer = nn.ModuleList([seqTransDecoderLayer1, seqTransDecoderLayer2, seqTransDecoderLayer3, seqTransDecoderLayer4,
                                                  seqTransDecoderLayer5, seqTransDecoderLayer6, seqTransDecoderLayer7, seqTransDecoderLayer8])
            self.decoder = TransformerEncoder(seqTransDecoderLayer)

        self.finalLinear = nn.Linear(num_channels, args.smpl_dim+9)
        self.bodyFinalLinear = nn.Linear(num_channels, args.smpl_dim+3)
        self.objFinalLinear = nn.Linear(num_channels, 9)
        self.bodyFutureEmbedding = nn.Parameter(torch.FloatTensor(args.future_len, 1, num_channels)) 
        self.bodyFutureEmbedding.data.uniform_(-1,1)
        self.objFutureEmbedding = nn.Parameter(torch.FloatTensor(args.future_len, 1, num_channels)) 
        self.objFutureEmbedding.data.uniform_(-1,1)

    def mask_cond(self, cond, force_mask=False):
        t, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.args.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.args.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def _get_embeddings(self, data, device=None):
        if device:
            body_pose = torch.cat([frame['smplfit_params']['pose'][:, :66].unsqueeze(0) for frame in data['frames']], dim=0).float().to(device) # TxBxDb
            body_trans = torch.cat([frame['smplfit_params']['trans'].unsqueeze(0) for frame in data['frames']], dim=0).float().to(device)  # TxBx3
            obj_angles = torch.cat([frame['objfit_params']['angle'].unsqueeze(0) for frame in data['frames']], dim=0).float().to(device)  # TxBx3
            obj_trans = torch.cat([frame['objfit_params']['trans'].unsqueeze(0) for frame in data['frames']], dim=0).float().to(device)  # TxBx3
            obj_points = data['obj_points'][:, :, :3].float().to(device)  # TxBxPx7
        else:
            body_pose = torch.cat([frame['smplfit_params']['pose'][:, :66].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBxDb
            body_trans = torch.cat([frame['smplfit_params']['trans'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBx3
            obj_angles = torch.cat([frame['objfit_params']['angle'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBx3
            obj_trans = torch.cat([frame['objfit_params']['trans'].unsqueeze(0) for frame in data['frames']], dim=0).float() # TxBx3
            obj_points = data['obj_points'][:, :, :3].float()

        T, B, _ = body_pose.shape
        pc_embedding = torch.cat([obj_points, obj_points.norm(dim=2, keepdim=True)], dim=2).unsqueeze(0)
        pc_embedding = self.pcEmbedding(pc_embedding).view(1, B, -1)
        body_pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose.view(T, B, -1, 3))).view(T, B, -1)
        obj_angles = matrix_to_rotation_6d(axis_angle_to_matrix(obj_angles.view(T, B, -1, 3))).view(T, B, -1)
        gt = torch.cat([body_pose, body_trans, obj_angles, obj_trans], dim=2)
        body, obj = torch.cat([body_pose, body_trans], dim=2), torch.cat([obj_angles, obj_trans], dim=2)

        body = self.bodyEmbedding(body[:self.args.past_len])
        obj = self.objEmbedding(obj[:self.args.past_len])
        embedding = body + obj + pc_embedding
        embedding = self.PositionalEmbedding(embedding)
        embedding = self.encoder(embedding)

        return embedding, gt


    def _decode(self, x, time_embedding, y=None):
        body, obj = torch.split(x, self.args.smpl_dim+3, dim=2)
        body = self.bodyEmbedding(body)
        obj = self.objEmbedding(obj)
        decoder_input = body + obj + time_embedding
        decoder_input = self.PositionalEmbedding(decoder_input)
        decoder_output = self.decoder(tgt=decoder_input, memory=y)

        body = self.bodyFinalLinear(decoder_output)
        obj = self.objFinalLinear(decoder_output)
        pred = torch.cat([body, obj], dim=2)
        return pred

    def forward(self, x, timesteps, y=None):
        time_embedding = self.embedTimeStep(timesteps)
        x = x.squeeze(1).permute(2, 0, 1).contiguous()
        if y is not None:
            y = self.mask_cond(y['cond'])
        x_0 = self._decode(x, time_embedding, y)
        x_0 = x_0.permute(1, 2, 0).unsqueeze(1).contiguous()
        return x_0

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.weight_v,
    )

def create_model_and_diffusion(args):
    model = MDM(args)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion