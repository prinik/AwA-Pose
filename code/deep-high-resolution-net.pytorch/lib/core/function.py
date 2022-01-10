# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy, accuracy_bbox
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images, save_debug_images_w_bbox


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        #_, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy())
        bbox = meta['bbox'].numpy()
        bbox_w = bbox[:, 1,0] - bbox[:, 0,0]
        bbox_h = bbox[:, 1,1] - bbox[:, 0,1]

        diagonal = np.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)        
        _, avg_acc, cnt, pred = accuracy_bbox(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy(), scale = None, thr=0.001*diagonal)
        
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_old(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, thresh = 0.0005, save_pickle=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    all_orig_boxes = np.zeros((num_samples, 2,2))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    ind_accs = []
    
    animal_list = [
     'antelope',
     'bobcat',
     'buffalo',
     'chihuahua',
     'collie',
     'cow',
     'dalmatian',
     'deer',
     'elephant',
     'fox',
     'german+shepherd',
     'giant+panda',
     'giraffe',
     'grizzly+bear',
     'hippopotamus',
     'horse',
     'leopard',
     'lion',
     'moose',
     'otter',
     'ox',
     'persian+cat',
     'pig',
     'polar+bear',
     'rabbit',
     'raccoon',
     'rhinoceros',
     'sheep',
     'siamese+cat',
     'skunk',
     'squirrel',
     'tiger',
     'weasel',
     'wolf',
     'zebra']
    
    animal_ids = list(range(8,36))
    crnt_animal_id = 8
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            
            if crnt_animal_id > 0 and animal_list[crnt_animal_id - 1] not in meta['image'][0]:
                continue
                
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            bbox = meta['bbox'].numpy()

            #import ipdb; ipdb.set_trace()
            #exit(0)             
            
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            #w_h_original = (s[:, [1, 0]] * 200.0) / 1.25
            #diagonal = np.sqrt(s[:, 0] * s[:, 0] + s[:, 1] * s[:, 1])
            
            bbox_w = bbox[:, 1,0] - bbox[:, 0,0]
            bbox_h = bbox[:, 1,1] - bbox[:, 0,1]
            
            diagonal = np.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)
            ind_acc, avg_acc, cnt, pred = accuracy_bbox(output.cpu().numpy(),
                                             target.cpu().numpy(), scale = s[:, [1, 0]], thr=thresh*diagonal) # swapping (w,h) to (h,w)

            ind_accs.append(ind_acc[1:])
            #print(avg_acc)
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            
            #saving pickles
            #if save_pickle:
            #    save_pred_pickle(config, input, meta, target, pred*4, output, prefix)
            idx += num_images

            if False:#i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                #save_debug_images(config, input, meta, target, pred*4, output, prefix)
                save_debug_images_w_bbox(config, input, meta, target, pred*4, output, prefix)

                #import ipdb; ipdb.set_trace()
                #exit(0)                   

        if False:

            name_values, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME
            
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, model_name)
            else:
                _print_name_value(name_values, model_name)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_scalar(
                    'valid_loss',
                    losses.avg,
                    global_steps
                )
                writer.add_scalar(
                    'valid_acc',
                    acc.avg,
                    global_steps
                )
                
                if isinstance(name_values, list):
                    for name_value in name_values:
                        writer.add_scalars(
                            'valid',
                            dict(name_value),
                            global_steps
                        )
                else:
                    writer.add_scalars(
                        'valid',
                        dict(name_values),
                        global_steps
                    )
                writer_dict['valid_global_steps'] = global_steps + 1
        else:
            perf_indicator = 100

    #return perf_indicator
    print('Average PCK @ '+ str(thresh) +'= ', acc.avg)
    ind_acc = [0]*len(ind_acc[1:])
    ind_acc_count = [0]*len(ind_acc)
    
    for ind_batch in ind_accs:
        for i, val in enumerate(ind_batch):
            if val >= 0:
                ind_acc_count[i] += 1
                ind_acc[i] += val
    
    for i in range(len(ind_acc)):
        
        #import ipdb; ipdb.set_trace()
        #exit(0)
        if ind_acc_count[i] == 0:
            ind_acc[i] = -1
        else:
            ind_acc[i] = ind_acc[i] / float(ind_acc_count[i])
     
        
    # import ipdb; ipdb.set_trace()
    # exit(0)      
    print("Independent keypoint accuracy: ",ind_acc)
    with open('Individual_Keypoint_Accuracy.txt', 'w') as f:
        for item in ind_acc:
            f.write("%s\n" % item)
    
    return acc.avg


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, thresh = 0.0005, save_pickle=True):

    
    animal_list = [
     'antelope',
     'bobcat',
     'buffalo',
     'chihuahua',
     'collie',
     'cow',
     'dalmatian',
     'deer',
     'elephant',
     'fox',
     'german+shepherd',
     'giant+panda',
     'giraffe',
     'grizzly+bear',
     'hippopotamus',
     'horse',
     'leopard',
     'lion',
     'moose',
     'otter',
     'ox',
     'persian+cat',
     'pig',
     'polar+bear',
     'rabbit',
     'raccoon',
     'rhinoceros',
     'sheep',
     'siamese+cat',
     'skunk',
     'squirrel',
     'tiger',
     'weasel',
     'wolf',
     'zebra',
     'all']
    
    animal_ids = list(range(1,36))
    #crnt_animal_id = 8
    
    for crnt_animal_id in animal_ids:

        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        num_samples = len(val_dataset)
        all_preds = np.zeros(
            (num_samples, config.MODEL.NUM_JOINTS, 3),
            dtype=np.float32
        )
        all_boxes = np.zeros((num_samples, 6))
        all_orig_boxes = np.zeros((num_samples, 2,2))
        image_path = []
        filenames = []
        imgnums = []
        idx = 0
        ind_accs = []

        with torch.no_grad():
            end = time.time()
            for i, (input, target, target_weight, meta) in enumerate(val_loader):
                # compute output
                
                # import ipdb; ipdb.set_trace()
                # exit(0)                  
                if crnt_animal_id > 0 and not meta['image'][0].split('/')[-1].startswith(animal_list[crnt_animal_id - 1]):
                    continue
                    
                outputs = model(input)
                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs

                if config.TEST.FLIP_TEST:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                               val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
                bbox = meta['bbox'].numpy()           
                
                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                #w_h_original = (s[:, [1, 0]] * 200.0) / 1.25
                #diagonal = np.sqrt(s[:, 0] * s[:, 0] + s[:, 1] * s[:, 1])
                
                bbox_w = bbox[:, 1,0] - bbox[:, 0,0]
                bbox_h = bbox[:, 1,1] - bbox[:, 0,1]
                
                diagonal = np.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)
                ind_acc, avg_acc, cnt, pred = accuracy_bbox(output.cpu().numpy(),
                                                 target.cpu().numpy(), scale = s[:, [1, 0]], thr=thresh*diagonal) # swapping (w,h) to (h,w)

                ind_accs.append(ind_acc[1:])
                #print(avg_acc)
                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                
                #saving pickles
                #if save_pickle:
                #    save_pred_pickle(config, input, meta, target, pred*4, output, prefix)
                idx += num_images

                if avg_acc < 0.35:#i % config.PRINT_FREQ == 0:
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                              i, len(val_loader), batch_time=batch_time,
                              loss=losses, acc=acc)
                    logger.info(msg)

                    prefix = '{}_{}'.format(
                        os.path.join(output_dir, 'bad', animal_list[crnt_animal_id - 1]), i
                    )
                    #save_debug_images(config, input, meta, target, pred*4, output, prefix)
                    # output heatmap size becomes 4 times smaller than the input image
                    save_debug_images_w_bbox(config, input, meta, target, pred*4, output, prefix)

                    # import ipdb; ipdb.set_trace()
                    # exit(0)                   

            if False:

                name_values, perf_indicator = val_dataset.evaluate(
                    config, all_preds, output_dir, all_boxes, image_path,
                    filenames, imgnums
                )

                model_name = config.MODEL.NAME
                
                if isinstance(name_values, list):
                    for name_value in name_values:
                        _print_name_value(name_value, model_name)
                else:
                    _print_name_value(name_values, model_name)

                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['valid_global_steps']
                    writer.add_scalar(
                        'valid_loss',
                        losses.avg,
                        global_steps
                    )
                    writer.add_scalar(
                        'valid_acc',
                        acc.avg,
                        global_steps
                    )
                    
                    if isinstance(name_values, list):
                        for name_value in name_values:
                            writer.add_scalars(
                                'valid',
                                dict(name_value),
                                global_steps
                            )
                    else:
                        writer.add_scalars(
                            'valid',
                            dict(name_values),
                            global_steps
                        )
                    writer_dict['valid_global_steps'] = global_steps + 1
            else:
                perf_indicator = 100

        #return perf_indicator
        print('Average PCK @ '+ str(thresh) +'= ', acc.avg)
        ind_acc = [0]*len(ind_acc[1:])
        ind_acc_count = [0]*len(ind_acc)
        
        for ind_batch in ind_accs:
            for i, val in enumerate(ind_batch):
                if val >= 0:
                    ind_acc_count[i] += 1
                    ind_acc[i] += val
        
        for i in range(len(ind_acc)):
            
            #import ipdb; ipdb.set_trace()
            #exit(0)
            if ind_acc_count[i] == 0:
                ind_acc[i] = -1
            else:
                ind_acc[i] = ind_acc[i] / float(ind_acc_count[i])
         
            
        # import ipdb; ipdb.set_trace()
        # exit(0)      
        print("Independent keypoint accuracy of " + animal_list[crnt_animal_id - 1] +": ",ind_acc)
        with open('Individual_Keypoint_Accuracy_'+ animal_list[crnt_animal_id - 1] +'.txt', 'w') as f:
            for item in ind_acc:
                f.write("%s\n" % item)
    
    return acc.avg


def generate_predited_keypoints_and_vis(config, img, model, output_dir,
             tb_log_dir):
    
    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    all_orig_boxes = np.zeros((num_samples, 2,2))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    ind_accs = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output            
                
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            bbox = meta['bbox'].numpy()           
            
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            #w_h_original = (s[:, [1, 0]] * 200.0) / 1.25
            #diagonal = np.sqrt(s[:, 0] * s[:, 0] + s[:, 1] * s[:, 1])
            
            bbox_w = bbox[:, 1,0] - bbox[:, 0,0]
            bbox_h = bbox[:, 1,1] - bbox[:, 0,1]
            
            diagonal = np.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)
            ind_acc, avg_acc, cnt, pred = accuracy_bbox(output.cpu().numpy(),
                                             target.cpu().numpy(), scale = s[:, [1, 0]], thr=thresh*diagonal) # swapping (w,h) to (h,w)

            ind_accs.append(ind_acc[1:])
            #print(avg_acc)
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            
            #saving pickles
            #if save_pickle:
            #    save_pred_pickle(config, input, meta, target, pred*4, output, prefix)
            idx += num_images


            save_debug_images_w_bbox(config, input, meta, target, pred*4, output, prefix)

            #import ipdb; ipdb.set_trace()
            #exit(0)                   




# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
