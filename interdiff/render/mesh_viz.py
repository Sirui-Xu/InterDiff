# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np
import torch
import trimesh
import pyrender
import math
from render.mesh_utils import MeshViewer
from data.utils import colors,bodypart2color,marker2bodypart67
import imageio

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c

def visualize_body_obj(body_verts, body_face, obj_verts, obj_face, past_len=0, pcd=None, pcd_contact=None, multi_col=None, text="",
                       multi_angle=True, h=512, w=512, bg_color='white',
                       save_path=None, fig_label=None, use_hydra_path=True, sample_rate=1):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )
 
    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    vis_mar = True if pcd is not None else False
    seqlen = len(body_verts)

    if isinstance(pcd, np.ndarray):
        pcd = torch.from_numpy(pcd)

    if vis_mar:
        pcd = pcd.reshape(seqlen, -1, 3).to('cpu')
        # if len(pcd.shape) == 3:
        #     pcd = pcd.unsqueeze(0)

    mesh_rec = -body_verts
    obj_mesh_rec = -obj_verts
    body_face = body_face[:, ::-1]
    obj_face = obj_face[:, ::-1]
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    mesh_rec[:, :, 1] -= height_offset
    obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2
    if vis_mar:
        pcd[:, :, 1] -= height_offset

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy, 
                    use_offscreen=True,
                    bg_color=bg_color)
                #ground_height=(mesh_rec.detach().cpu().numpy()[0, 0, 6633, 2] - 0.01))
    # ground plane has a bug if we want batch_size to work
    mv.render_wireframe = False
    
    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 4 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):
        # Rx = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        # Ry = trimesh.transformations.rotation_matrix(math.radians(180), [0, 1, 0])

        if i <= past_len:
            obj_mesh_color = np.tile(c2rgba(colors['grey']), (obj_mesh_rec.shape[1], 1))
        else:
            obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)
        # obj_m_rec.apply_transform(Rx)
        # obj_m_rec.apply_transform(Ry)

        if i <= past_len:
            mesh_color = np.tile(c2rgba(colors['light_grey']), (mesh_rec.shape[1], 1))
        else:
            mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))
            
        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)
        all_meshes = []
        # m_rec.apply_transform(Rx)
        # m_rec.apply_transform(Ry)
        if vis_mar:
            m_pcd = trimesh.points.PointCloud(pcd[i])
            pcd_bodyparts = tobodyparts(m_pcd, i <= past_len)
            for bp, m_bp in pcd_bodyparts.items():
                all_meshes.append(m_bp)

        all_meshes = all_meshes + [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(3):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                if vis_mar:
                    m_pcd.apply_transform(Ry)
                    pcd_bodyparts = tobodyparts(m_pcd, i <= past_len)
                    for bp, m_bp in pcd_bodyparts.items():
                        all_meshes.append(m_bp)
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                video_views.append(mv.render())
            video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1), np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)

        video[i] = video_i

    if save_path is not None:
        imageio.mimsave(save_path, list(np.squeeze(video).astype(np.uint8)), fps=30 // sample_rate)

    del mv
    return np.transpose(np.squeeze(video), (0, 3, 1, 2)).astype(np.uint8)

def tobodyparts(m_pcd, past=False):
    m_pcd = np.array(m_pcd.vertices)
    # after trnaofrming poincloud visualize for each body part separately
    pcd_bodyparts = dict()
    for bp, ids in marker2bodypart67.items():
        points = m_pcd[ids]
        tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
        tfs[:, :3, 3] = points
        col_sp = trimesh.creation.uv_sphere(radius=0.01)

        # debug markers, maybe delete it
        # if bp == 'special':
        #     col_sp = trimesh.creation.uv_sphere(radius=0.03)

        if past:
            col_sp.visual.vertex_colors = c2rgba(colors["black"])
        else:
            col_sp.visual.vertex_colors = c2rgba(colors[bodypart2color[bp]])

        pcd_bodyparts[bp] = pyrender.Mesh.from_trimesh(col_sp, poses=tfs)
    return pcd_bodyparts