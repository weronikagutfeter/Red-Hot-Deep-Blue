import math
import sys
import time
import torch
from torch.optim import lr_scheduler
import os
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm
from iphd.evaluate import eval_ap
import pickle
from torchvision import transforms as pytransforms

import utils

converter = pytransforms.ToPILImage()

def train_one_epoch(epoch, model, data_loader, tbwriter, device, optimizer, print_freq, max_iterations):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    cc = 0
    ave_losses = None
    ave_losses_c = 0
    for images, imfilenames, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if(ave_losses is None):
            ave_losses = dict()
            ave_losses['loss'] = loss_value
            for kvp in loss_dict_reduced.items():
                ave_losses[kvp[0]] = kvp[1].item()
            ave_losses_c = 1
        else:
            ave_losses['loss'] += loss_value
            for kvp in loss_dict_reduced.items():
                ave_losses[kvp[0]] += kvp[1].item()
            ave_losses_c += 1

        if(cc>max_iterations):
            break
        cc = cc+1

    if(ave_losses is not None):
        for kvp in ave_losses.items():
            tbwriter.add_scalar('Training/' + kvp[0], kvp[1]/ave_losses_c, epoch)

def train_one_epoch_with_dualsource(epoch, model,  data_loader_dd, tbwriter, device, optimizer, print_freq, max_iterations, second_model=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader_dd) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    cc = 0
    ave_losses = None
    ave_losses_c = 0
    for images, imfilenames, gt_targets in metric_logger.log_every(data_loader_dd, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in gt_targets]

        if (len(targets) < 1):
            continue

        if(second_model is None):
            # samples = [torch.cat((a, a, b.to(device)), 0) for a, b in zip(images, dd_images)]

            loss_dict = model(images, targets)
        # else:
        #     loss_dict1 = model(images,targets)
        #     loss_dict2 = model(dd_images,targets)
        del targets

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if(ave_losses is None):
            ave_losses = dict()
            ave_losses['loss'] = loss_value
            for kvp in loss_dict_reduced.items():
                ave_losses[kvp[0]] = kvp[1].item()
            ave_losses_c = 1
        else:
            ave_losses['loss'] += loss_value
            for kvp in loss_dict_reduced.items():
                ave_losses[kvp[0]] += kvp[1].item()
            ave_losses_c += 1

        if(cc>max_iterations):
            break
        cc = cc+1

    if(ave_losses is not None):
        for kvp in ave_losses.items():
            tbwriter.add_scalar('Training/' + kvp[0], kvp[1]/ave_losses_c, epoch)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def iphd_evaluation(boxes,target_boxes):
    AP = {}
    AP[25] = eval_ap(boxes, target_boxes, 25 / 100.)
    mAP = 0
    for iou in range(50,95,5):
        AP[iou] = eval_ap(boxes, target_boxes, iou / 100.)
        mAP += AP[iou]
    mAP = mAP/10

    # for iou in [25, 50, 75]:
    #     AP[iou] = eval_ap(boxes, target_boxes, iou / 100.)
    n_frames = len(boxes.keys())
    sum_boxes = 0.0
    for b in boxes.values():
        sum_boxes+=len(b)
    n_ave_boxes = sum_boxes/n_frames
    sum_target_boxes = 0.0
    for t in target_boxes.values():
        sum_target_boxes += len(t)
    n_ave_target_boxes = sum_target_boxes/n_frames
    return AP, n_frames, n_ave_boxes, n_ave_target_boxes, mAP

def draw_evaluation(outfilename, I, boxes, target_boxes, scores=None, hot=True, preprocess=True):
    DM = np.array(I)
    if(preprocess):
        if not (hot):
            max_depth = 50000
            DM[DM > max_depth] = max_depth
            DM = (DM / np.max(DM))
            DM = np.uint8(cm.bone(1 - DM) * 255)
        else:
            DM[DM < 28315] = 28315  # clip at 283.15 K (or 10 ºC)
            DM[DM > 31315] = 31315  # clip at 313.15 K (or 40 ºC)
            DM = ((DM - 28315.) / (31315 - 28315))
            DM = np.uint8(cm.hot(DM) * 255)
    # else:
        # if not (hot):
        #     DM = np.uint8(cm.bone(1 - DM/255) * 255)
        # else:
        #     DM = np.uint8(cm.hot(DM/255) * 255)
    IM = Image.fromarray(DM)

    imd = ImageDraw.Draw(IM)
    if boxes is not None:
        for b,box in enumerate(boxes):
            recshape = ((box[0], box[1]), (box[2], box[3]))
            imd.rectangle(recshape, outline="blue")#, width=3)
            if (scores is not None):
                imd.text((box[0], box[1]), "{:.2f}".format(scores[b]))
    if target_boxes is not None:
        for tbox in target_boxes:
            recshape = ((tbox[0], tbox[1]), (tbox[2], tbox[3]))
            if (hot):
                imd.rectangle(recshape, outline="green")#, width=2)
            else:
                imd.rectangle(recshape, outline="red")#, width=3)

    IM.save(outfilename)


@torch.no_grad()
def evaluate(epoch, model, data_loader, tbwriter, device, max_iterations=None, header_prefix='Validation'):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '{}:'.format(header_prefix)

    cc=1
    boxes_dict = dict()
    target_boxes_dict = dict()

    if(max_iterations is None):
        max_iterations = len(data_loader)

    for batch in metric_logger.log_every(data_loader, 100, header):
        # if(fusion):
        #     images, _, imfilenames, targets = batch
        # else:
        images, imfilenames, targets = batch
        if (cc > max_iterations):
            break
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        outputs = model(images)

        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]


        oo = outputs[0]
        boxes = [t.numpy() for t in oo['boxes']]
        scores = [t.numpy() for t in oo['scores']]

        tt = targets[0]
        target_boxes = [t.cpu().numpy() for t in tt['boxes']]

        imname = imfilenames[0]

        target_boxes_dict[imname] = [np.concatenate((t,np.array([1.0,0]))) for t in target_boxes]
        boxes_dict[imname] = [np.concatenate((b, np.array([s, 0]))) for b,s in zip(boxes,scores)]


        cc = cc + 1
    ap, n_frames, n_ave_boxes, n_ave_target_boxes, mAP = iphd_evaluation(boxes_dict, target_boxes_dict)
    print("\tAP-25: {:.5f} AP-50: {:.5f} AP-75: {:.5f} COCO mAP: {:.5f} Ave n boxes: {} Ave n target boxes: {} in {} frames".format(ap[25],ap[50],ap[75],mAP,
                                                                                                                   n_ave_boxes,n_ave_target_boxes,n_frames))

    tbwriter.add_scalar(header_prefix+'/AP-25', ap[25], epoch)
    tbwriter.add_scalar(header_prefix+'/AP-50', ap[50], epoch)
    tbwriter.add_scalar(header_prefix+'/AP-75', ap[75], epoch)
    tbwriter.add_scalar(header_prefix+'/mAP', mAP, epoch)
    tbwriter.add_scalar(header_prefix+'/nboxes', n_ave_boxes, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)


    torch.set_num_threads(n_threads)
    return ap[50], ap[75], mAP

@torch.no_grad()
def draw_one_epoch(epoch, model, data_loader, tempdir, device, hot):
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    cc=1

    for images, imfilenames, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        oo = outputs[0]
        boxes = [t.numpy() for t in oo['boxes']]

        tt = targets[0]
        target_boxes = [t.cpu().numpy() for t in tt['boxes']]

        scores = [t.cpu().numpy() for t in oo['scores']]

        imname = imfilenames[0]

        outfilename = os.path.join(tempdir,'e{}_'.format(epoch)+os.path.basename(imname))
        IM = Image.open(imname)
        draw_evaluation(outfilename,IM,boxes,target_boxes, scores=scores, hot=hot)

        cc = cc + 1



@torch.no_grad()
def save_one_epoch(model, data_loader, outputfile, device):

    model.eval()

    objects_dict = dict()

    k = 0
    for images, imfilenames, targets in data_loader:
        print('\r{}/{}'.format(k,len(data_loader)),end='')
        k=k+1
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        for image, imname, output in zip(images, imfilenames,outputs):
            values = []
            labelname = os.path.splitext(os.path.basename(imname))[0]
            w = image.shape[2]
            h = image.shape[1]
            for box, score in zip(output["boxes"], output["scores"]):
                box = box.numpy()
                outbox = np.array([(box[0]+box[2])/(2*w),(box[1]+box[3])/(2*h),(box[2]-box[0])/w,(box[3]-box[1])/h])
                score = score.numpy()
                values.append(np.concatenate((outbox,[score])))
            objects_dict[labelname] = values


    with open(outputfile, 'wb') as f:
        pickle.dump(objects_dict, f, 2)
