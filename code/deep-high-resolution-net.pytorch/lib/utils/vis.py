# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds


def save_demo_images(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=1, padding=0):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    
    skeleton = [(10, 5), (10, 11), (5, 6), (6, 7), (11, 12), (0, 5), (0, 10), (1, 2), (3, 4), (6, 15), (11, 15), (15, 16), (16, 19), (19, 21), (21, 20), (20, 22), (22, 23), (2, 17), (17, 18), (26, 25), (25, 24), (28, 29), (29, 27), (31, 30), (30, 32), (34, 35), (35, 33), (36, 27), (36, 24), (36, 32), (36, 33), (37, 27), (37, 33), (38, 32), (38, 24), (38, 19), (38, 20), (37, 19), (37, 20), (18, 24), (18, 27), (6, 8), (8, 9), (11, 13), (13, 14)]

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                    
            for bone in skeleton:
                if joints_vis[bone[0]][0] and joints_vis[bone[1]][0]:
                    point1 = (int(joints[bone[0]][0]), int(joints[bone[0]][1]))
                    point2 = (int(joints[bone[1]][0]), int(joints[bone[1]][1]))
                    cv2.line(ndarr, point1, point2, [0, 255, 0], 2)
                    
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    
    skeleton = [(10, 5), (10, 11), (5, 6), (6, 7), (11, 12), (0, 5), (0, 10), (1, 2), (3, 4), (6, 15), (11, 15), (15, 16), (16, 19), (19, 21), (21, 20), (20, 22), (22, 23), (2, 17), (17, 18), (26, 25), (25, 24), (28, 29), (29, 27), (31, 30), (30, 32), (34, 35), (35, 33), (36, 27), (36, 24), (36, 32), (36, 33), (37, 27), (37, 33), (38, 32), (38, 24), (38, 19), (38, 20), (37, 19), (37, 20), (18, 24), (18, 27), (6, 8), (8, 9), (11, 13), (13, 14)]

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                    
            for bone in skeleton:
                if joints_vis[bone[0]][0] and joints_vis[bone[1]][0]:
                    point1 = (int(joints[bone[0]][0]), int(joints[bone[0]][1]))
                    point2 = (int(joints[bone[1]][0]), int(joints[bone[1]][1]))
                    cv2.line(ndarr, point1, point2, [0, 255, 0], 2)
                    
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_joints_n_bbox(batch_image, bboxs, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    
    #import ipdb; ipdb.set_trace()
    #exit(0)      
    
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            bbox = bboxs[k]
            
            x1,y1 = bbox[0,0],bbox[0,1]
            x2,y2 = bbox[1,0],bbox[1,1]
            x1 = x * width + padding + x1
            y1 = y * height + padding + y1
            
            x2 = x * width + padding + x2
            y2 = y * height + padding + y2
            
            start_point = (int(x1),int(y1))
            end_point = (int(x2),int(y2))
            
            cv2.rectangle(ndarr, start_point, end_point, [255, 0, 0], 2)

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)
    #import ipdb; ipdb.set_trace()
    #exit(0)     
    
    
    
def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )

def save_debug_images_w_bbox(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints_n_bbox(
            input, meta['bbox'], meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
