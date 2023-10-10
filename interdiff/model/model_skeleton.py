import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix
from model.layers import PositionalEncoding, TimestepEmbedder, TransformerEncoder, TransformerDecoder
from model.sublayers import TransformerDecoderLayerQaN, TransformerEncoderLayerQaN

class MDM(nn.Module):
    # input:
    # 21 human body keypoints 
    # 12 object keypoints
    # 7-D object pose
    # output:
    # predicted  
    def __init__(self, args):
        super(MDM, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        self.bodyEmbedding = nn.Linear(args.smpl_dim, num_channels)
        self.shapeEmbedding = nn.Linear(args.num_points*3, num_channels)
        self.objEmbedding = nn.Linear(args.num_points*3, num_channels)
        self.PositionalEmbedding = PositionalEncoding(d_model=num_channels, dropout=args.dropout)
        self.embedTimeStep = TimestepEmbedder(num_channels, self.PositionalEmbedding)
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

        self.bodyFinalLinear = nn.Linear(num_channels, args.smpl_dim)
        self.objFinalLinear = nn.Linear(num_channels, 7)

    def mask_cond(self, cond, force_mask=False):
        t, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.args.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.args.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def _get_embeddings(self, body_gt, obj_gt, pose_gt, zero_pose_obj):
        # input: T,B,21,3 body joints
        # T,B,12,3 obj keypoints
        # T,B,7 obj pose (which we don't use)
        # zero_pose: B,12,3
        T,B,N_joints, _ = body_gt.shape
        N_points = obj_gt.shape[2]
        body_gt = body_gt.view(T,B,N_joints*3)
        obj_gt = obj_gt.view(T,B,N_points*3)

        obj_shape_embedding = self.shapeEmbedding(zero_pose_obj[None].view(1,B,N_points*3))

        gt = torch.cat([body_gt,obj_gt,pose_gt], dim=2)# T,B,N_gt

        body = self.bodyEmbedding(body_gt[:self.args.past_len])
        obj = self.objEmbedding(obj_gt[:self.args.past_len])
        embedding = body + obj + obj_shape_embedding
        # T,B,embed_dim
        embedding = self.PositionalEmbedding(embedding)
        embedding = self.encoder(embedding)

        return embedding, gt


    def calc_obj_pred(self, pose_pred, zero_pose_obj):
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

    def _decode(self, x, time_embedding, y=None, zero_pose_obj=None):
        # x: T,B,N_gt
        assert zero_pose_obj is not None
        T,B,N_gt = x.shape

        body, obj, pose = torch.split(x, [self.args.num_joints*3, self.args.num_points*3,7], dim=2)
        body = self.bodyEmbedding(body)
        obj = self.objEmbedding(obj)

        decoder_input = body + obj + time_embedding
        decoder_input = self.PositionalEmbedding(decoder_input)
        decoder_output = self.decoder(tgt=decoder_input, memory=y)

        body = self.bodyFinalLinear(decoder_output)
        obj_pose = self.objFinalLinear(decoder_output)
        obj = self.calc_obj_pred(obj_pose, zero_pose_obj).view(T,B,-1)
        pred = torch.cat([body, obj, obj_pose], dim=2)
        return pred

    def forward(self, x, timesteps,zero_pose_obj, y=None):
        time_embedding = self.embedTimeStep(timesteps)
        x = x.squeeze(1).permute(2, 0, 1).contiguous()
        if y is not None:
            y = self.mask_cond(y['cond'])
        x_0 = self._decode(x, time_embedding, y, zero_pose_obj=zero_pose_obj)
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