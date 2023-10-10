from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

import matplotlib.image as mpimg

from matplotlib.colors import to_rgba
connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (2, 5), (5, 6), (6, 7), (7, 8), 
                             (2, 9), (9, 10), (10, 11), (11, 12),
                             (0, 13), (13, 14), (14, 15), 
                             (0, 16), (16, 17), (17, 18)]

obj_connects = {'chair4': [(1,2),(1,4),(2,4),(1,0),(0,2),(0,5),(5,7),(0,10),(2,11),(4,9),(1,8),(2,3),(5,3),(4,6),(0,6),\
              (0,7),(2,7),(3,7)], 'box2':[(2,11),(2,5),(9,11),(1,0),(1,7),(8,10),(3,4),(4,9),(3,8),(7,8),(1,11),(3,5),(6,2),(3,6),\
               (2,0),(4,10),(6,8),(1,2),(7,10),(7,0),(4,5),(5,11),(0,6),(6,7),(1,9),(9,10),(5,9),(7,9)], 'board':
               [(3,6),(6,5),(3,9),(5,9),(5,1),(1,4),(2,4),(1,7),(0,7),(0,11),(11,10),(8,10),(2,8),(2,9)],
               'chair2': [(4,9),(2,11),(1,8),(0,10),(0,1),(1,4),(2,4),(2,3),(3,5),(0,2),(0,5),(7,3),(7,5),(7,0),(7,2),
               (0,6),(6,1),(6,2),(6,4)], 'box3':[(4,5),(5,9),(5,11),(2,5),(2,6),(2,0),(2,11),(9,4),(9,11),(9,1),(9,10),(1,0),(1,7),(0,6),
               (3,4),(3,5),(3,10),(3,8),(8,6),(8,7),(8,10),(3,6),(0,7),(1,11),(4,10),(10,7)],
               'table': [(0,2),(2,3),(3,4),(4,0),(0,1),(2,1),(1,10),(3,5),(2,5),(5,8),(4,6),(3,6),(6,7),(0,11),(4,11),(11,9)], 'chair':
               [(4,9),(2,11),(1,8),(0,10),(0,1),(1,4),(2,4),(2,3),(3,5),(0,2),(0,5),(7,3),(7,5),(7,0),(7,2),
               (0,6),(6,1),(6,2),(6,4)], 'box': [(4,5),(5,9),(5,11),(2,5),(2,6),(2,0),(2,11),(9,4),(9,11),(9,1),(9,10),(1,0),(1,7),(0,6),
               (3,4),(3,5),(3,10),(3,8),(8,6),(8,7),(8,10),(3,6),(0,7),(1,11),(4,10),(10,7)],'tripod':
               [(3,5),(4,6),(0,1),(7,10),(7,11),(9,7),(1,8),(4,8),(5,8),(8,2),(8,7),(7,10)]}
def visualize_skeleton(skeledonData, objPoint, save_dir = './test.gif'):
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection="3d")
    def updateHumanObj(frame, *fargs):

        ax.clear()
        bodyData, objPoint, scat = fargs
        z_points = bodyData[frame, :, 2] #* -1.0
        x_points = bodyData[frame, :, 0]
        y_points = bodyData[frame, :, 1]
        for connect in connections:
            a,b = connect
            ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color="b")
        ax.scatter3D(x_points, y_points, z_points, color="r")

        thisObjPoint = objPoint[frame].reshape((12,3))
        z_points = thisObjPoint[ :, 2] #* -1.0
        x_points = thisObjPoint[ :, 0] 
        y_points = thisObjPoint[ :, 1]
        ax.scatter3D(x_points, y_points, z_points, color="g")
        #'''
        ax.plot([0.0, 0.0],[-0.7,1.2,],[0.0,0.0], color="w")#b
        ax.plot([-1.2,1.2],[0.0,0.0,],[0.0,0.0], color="w")#r
        ax.plot([0.0, 0.0],[0.0,0.0,],[-1.2,1.2], color="w")#g
        # plt.axis('off')
        ax.set_axis_off()
        return ax


    def visualize(skeledonData, objPoint, startFrame = 0):
        bodyData = skeledonData
        lenFrame = skeledonData.shape[0]
        
        bodyData = bodyData.reshape((lenFrame, 21, 3))
        # ax.yaxis.set_label_position("top")
        ax.view_init(elev=117., azim=-88.)
        scat = ax.scatter(bodyData[0,:,0], bodyData[0,:,1], bodyData[0,:,2], c='r', marker = 'o',alpha=0.5, s=100)
        
        #time.sleep(.01)
        ani = animation.FuncAnimation(fig, updateHumanObj, frames= range(lenFrame), interval = 50, repeat_delay=100,
                                    fargs=(bodyData, objPoint, scat))
        ani.save(save_dir)

    visualize(skeledonData, objPoint)
    plt.clf()
    plt.close()


def visualize_skeleton_pred_gt(skeledonData, objPoint, skeledonData_gt, objPoint_gt, save_dir = './test.gif', elev=117., azim=-88., roll=0.0, obj_name='chair4', tmp_dir='./tmp'):
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection="3d")
    
    Path(tmp_dir).mkdir(exist_ok=True, parents=True)
    ax.view_init(elev=elev, azim=azim, roll=roll,vertical_axis='y')
    def updateHumanObj(frame, *fargs):

        ax.clear()
        ax.plot([0.0, 0.0],[-0.7,1.2,],[0.0,0.0], color="w")#b
        ax.plot([-1.2,1.2],[0.0,0.0,],[0.0,0.0], color="w")#r
        ax.plot([0.0, 0.0],[0.0,0.0,],[-1.2,1.2], color="w")#g
        # ax.plot([0.0, 0.0],[-0.49,0.7,],[0.0,0.0], color="w")#b
        # ax.plot([-0.7,0.7],[0.0,0.0,],[0.0,0.0], color="w")#r
        # ax.plot([0.0, 0.0],[0.0,0.0,],[-0.7,0.7], color="w")#g
        
        bodyData, objPoint, bodyData_gt, objPoint_gt = fargs
        obj_connections = obj_connects.get(obj_name,[])
        if frame>10:
            # visualize pred
            z_points = bodyData[frame, :, 2] #* -1.0
            x_points = bodyData[frame, :, 0]
            y_points = bodyData[frame, :, 1]
            p_human_color = to_rgba("seagreen")
            p_human_color = (*p_human_color[:3],0.8)
            for connect in connections:
                a,b = connect
                ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color=p_human_color)
            # ax.scatter3D(x_points, y_points, z_points, color="purple")

            thisObjPoint = objPoint[frame].reshape((12,3))
            z_points = thisObjPoint[ :, 2] #* -1.0
            x_points = thisObjPoint[ :, 0] 
            y_points = thisObjPoint[ :, 1]
            
            p_obj_color = to_rgba("salmon")
            p_obj_color = (*p_obj_color[:3],0.8)
            for connect in obj_connections:
                a,b = connect
                ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color=p_obj_color)
            # ax.scatter3D(x_points, y_points, z_points, color="r")

        # NOTE: plot gt
        if frame<objPoint_gt.shape[0]:
            gt_human_color = to_rgba("lightgrey")
            gt_human_color = (*gt_human_color[:3],0.8)
            z_points_gt = bodyData_gt[frame, :, 2] #* -1.0
            x_points_gt = bodyData_gt[frame, :, 0]
            y_points_gt = bodyData_gt[frame, :, 1]
            for connect in connections:
                a,b = connect
                ax.plot([x_points_gt[a], x_points_gt[b]],[y_points_gt[a],y_points_gt[b]],[z_points_gt[a],z_points_gt[b]], color=gt_human_color)#royalblue 
            # ax.scatter3D(x_points_gt, y_points_gt, z_points_gt, color="b")

            thisObjPoint = objPoint_gt[frame].reshape((12,3))
            
            z_points_gt = thisObjPoint[ :, 2] #* -1.0
            x_points_gt = thisObjPoint[ :, 0] 
            y_points_gt = thisObjPoint[ :, 1]
            gt_obj_color = to_rgba("dimgrey")
            gt_obj_color = (*gt_obj_color[:3],0.5)
            for connect in obj_connections:
                a,b = connect
                ax.plot([x_points_gt[a], x_points_gt[b]],[y_points_gt[a],y_points_gt[b]],[z_points_gt[a],z_points_gt[b]], color=gt_obj_color)#teal
        # ax.scatter3D(x_points, y_points, z_points, color="g")

        #'''

        ax.set_axis_off()
        return ax


    def visualize(skeledonData, objPoint, skeledonData_gt, objPoint_gt, startFrame = 0):
        real_save_dir = save_dir[:-4]+'.gif'
        bodyData = skeledonData
        lenFrame = skeledonData.shape[0]
        ax.view_init(elev=elev, azim=azim, roll=roll,vertical_axis='y')
        # gt: blue and green
        # pred: purple and red
        
        bodyData = bodyData.reshape((lenFrame, 21, 3))
        scat = ax.scatter(bodyData[0,:,0], bodyData[0,:,1], bodyData[0,:,2], c='b', marker = 'o',alpha=0.5, s=100)
        
        #time.sleep(.01)
        ani = animation.FuncAnimation(fig, updateHumanObj, frames= range(lenFrame), interval = 50, repeat_delay=100,
                                    fargs=(bodyData, objPoint, skeledonData_gt, objPoint_gt))
        ani.save(real_save_dir)
        
    def visualize_imgs(skeledonData, objPoint, skeledonData_gt, objPoint_gt):
        real_save_dir = save_dir[:-4]+'.png'
        total = 20
        steps = [3,7,11,15,19]
        images = []
        x_scale = 779
        y_scale = 779#dpi=200
        x_tight_amnt_left = 60*2
        x_tight_amnt_right = 60*2
        x_rescaled = x_scale - x_tight_amnt_left - x_tight_amnt_right
        y_tight_amnt_upper = 60*2
        y_tight_amnt_lower = 100*2
        y_rescaled = y_scale - y_tight_amnt_upper  - y_tight_amnt_lower
        
        
        for frame in steps:
            fargs=(skeledonData, objPoint,  skeledonData_gt, objPoint_gt)
            ax = updateHumanObj(frame, *fargs)
            tmppath = str(Path(tmp_dir)/f"{frame}.png")
            plt.savefig(tmppath, bbox_inches='tight',dpi=200)
            # print(mpimg.imread(tmppath).shape)
            images.append(mpimg.imread(tmppath)[y_tight_amnt_upper:y_tight_amnt_upper+y_rescaled,x_tight_amnt_left:x_tight_amnt_left+x_rescaled])
            # 480,640,3
            # 0-255

        # concat imgs
        big_img = np.zeros((y_rescaled, (x_rescaled*5),4))
        for i in range(len(steps)):
            big_img[:,x_rescaled*i:x_rescaled*(i+1)]=images[i]
        mpimg.imsave(real_save_dir, big_img)
        
    
        # output, an array of images at different timesteps
    # visualize_imgs(skeledonData, objPoint, skeledonData_gt, objPoint_gt)
    visualize(skeledonData, objPoint, skeledonData_gt, objPoint_gt)
    plt.clf()
    plt.close()