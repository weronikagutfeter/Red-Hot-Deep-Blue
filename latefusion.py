import os
import numpy as np
from PIL import Image, ImageDraw
import pickle
from matplotlib import cm
from iphd.evaluate import bbox_iou
from iphd.evaluate import eval_ap
from datetime import datetime
from eval import object_detection_testing

def nms(boxes, scores, iou_threshold = 0.6):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep



def dual_nms(boxes_depth,scores_depth,boxes_thermal,scores_thermal, iou_threshold = 0.6):
    res_boxes = []
    res_scores = []

    if(len(scores_depth)>0):
        ind_depth = np.argsort(scores_depth,axis=0)[::-1]
        boxes_depth=[boxes_depth[ii] for ii in ind_depth]
        scores_depth=[scores_depth[ii] for ii in ind_depth]
    else:
        return boxes_thermal, scores_thermal
    if(len(scores_thermal)>0):
        ind_thermal = np.argsort(scores_thermal, axis=0)[::-1]
        boxes_thermal = [boxes_thermal[ii] for ii in ind_thermal]
        scores_thermal = [scores_thermal[ii] for ii in ind_thermal]
    else:
        return boxes_depth,scores_depth

    for box_th, score_th in zip(boxes_thermal,scores_thermal):

        if(len(boxes_depth)>0):
            box_th = np.array(box_th)
            ious = bbox_iou(box_th[np.newaxis, :], np.asarray(boxes_depth))
            selected = list(filter(lambda x: ious[x] > iou_threshold, range(len(ious))))
            if(len(selected)>0):
                selected_scores = np.array([scores_depth[s] for s in selected])
                max_score = np.max(selected_scores)
                res_boxes.append(box_th)
                res_scores.append((score_th + max_score) / 2)
                boxes_depth = list(filterme(boxes_depth,selected))
                scores_depth = list(filterme(scores_depth,selected))
                # for s in selected:
                    # boxes_depth.pop(s)
                    # scores_depth.pop(s)
            elif(score_th>0):
                res_boxes.append(box_th)
                res_scores.append(0.6*score_th)
    for box_d, score_d in zip(boxes_depth, scores_depth):
        if (score_d > 0):
            res_boxes.append(box_d)
            res_scores.append(score_d*0.4)

    return res_boxes,res_scores

def filterme(a,toremove):
    for x in range(len(a)):
        if(x not in toremove):
            yield a[x]

def draw(label,boxes, scores, infile,outfile,hot,color):
    if os.path.isfile(infile):
        I = Image.open(infile)
        width, height = I.size

        DM = np.array(I)

        if not (hot):
            max_depth = 50000
            DM[DM > max_depth] = max_depth
            DM = (DM / np.max(DM))
            DM = np.uint8(cm.bone(1 - DM) * 255)
            width = 1280
            height = 720
        else:
            DM[DM < 28315] = 28315  # clip at 283.15 K (or 10 ºC)
            DM[DM > 31315] = 31315  # clip at 313.15 K (or 40 ºC)
            DM = ((DM - 28315.) / (31315 - 28315))
            DM = np.uint8(cm.hot(DM) * 255)
            width = 213
            height = 120
        IM = Image.fromarray(DM)

        imd = ImageDraw.Draw(IM)


        for c, dets in enumerate(boxes):
            if(scores[c]>0.2):
                recshape = [dets[0]*width, dets[1]*height,  dets[2]*width, dets[3]*height]
                imd.text((dets[0] * width, dets[1] * height), "{:.2f}".format(scores[c]))
                imd.rectangle(recshape, outline=color)  # , width=3)
        IM.save(outfile)

def merging(depth_modelpicklename, thermal_modelpicklename, outputfile, imdir_depth,  imdir_thermal, ground_truth, output_debug, opt, draw_debug=False):

    print("Opt is {}".format(opt))


    objects_dict = dict()

    thermal_predictions_pickle = thermal_modelpicklename.replace('.pkl','_predictions.pkl')
    if(not(os.path.exists(thermal_predictions_pickle))):
        object_detection_testing(thermal_modelpicklename, thermal_predictions_pickle, imdir_thermal, hot=True)

    depth_predictions_pickle = depth_modelpicklename.replace('.pkl', '_predictions.pkl')
    if (not (os.path.exists(depth_predictions_pickle))):
        object_detection_testing(depth_modelpicklename, depth_predictions_pickle, imdir_depth, hot=False)

    with open(depth_predictions_pickle, 'rb') as f:
        depth_dict = pickle.load(f)

        with open(thermal_modelpicklename.replace('.pkl','_predictions.pkl'), 'rb') as f:
            thermal_dict = pickle.load(f)

            k = 0
            for kvp_depth in depth_dict.items():
                label = kvp_depth[0]
                k = k + 1
                w = 1280
                h = 720
                boxes_d = [[b[0]-b[2]/2, b[1]-b[3]/2,b[0]+b[2]/2, b[1]+b[3]/2] for b in
                           kvp_depth[1]]
                scores_d = [b[4] for b in kvp_depth[1]]

                entry_thermal = thermal_dict[label]
                w = 213
                h = 120
                boxes_th = [[b[0]-b[2]/2, b[1]-b[3]/2,b[0]+b[2]/2, b[1]+b[3]/2] for b
                            in entry_thermal]
                scores_th = [b[4] for b in entry_thermal]


                if(opt=='vanilla'):
                    if(len(boxes_d)<1):
                        boxes = boxes_th
                        scores = scores_th
                    elif(len(boxes_th)<1):
                        boxes = boxes_d
                        scores = scores_d
                    else:
                        boxes = boxes_d + boxes_th
                        scores = scores_d + scores_th

                    if(len(boxes)>0):
                        boxes = np.array(boxes)
                        scores = np.array(scores)
                        inds = nms(boxes, scores)
                        res_boxes = boxes[inds,:]
                        res_scores = scores[inds]
                    else:
                        res_boxes = boxes
                        res_scores = scores
                elif(opt=='my'):

                    res_boxes,res_scores = dual_nms(boxes_d,scores_d,boxes_th,scores_th)
                else:
                    if (len(boxes_d) < 1):
                        boxes = boxes_th
                        scores = scores_th
                    elif (len(boxes_th) < 1):
                        boxes = boxes_d
                        scores = scores_d
                    else:
                        boxes = boxes_d + boxes_th
                        scores = scores_d + scores_th
                    res_boxes = np.array(boxes)
                    res_scores = np.array(scores)


                print('\r{}/{} {} depth boxes {} thermal boxes  {} nms boxes'.format(k, len(depth_dict.keys()),len(boxes_d),len(boxes_th),len(res_boxes)),end='')


                values = []
                for box, score in zip(res_boxes, res_scores):
                    outbox = np.array([(box[0] + box[2]) / 2 , (box[1] + box[3]) / 2, (box[2] - box[0]),
                                       (box[3] - box[1]) ])
                    values.append(np.concatenate((outbox, [score])))

                objects_dict[label] = values
                if(draw_debug):
                    infile = os.path.join(imdir_depth,label+'.png')

                    outfile = os.path.join(output_debug,label+'_nms.png')
                    draw(label,res_boxes,res_scores,infile,outfile,hot=False,color='green')
                    #
                    # outfile = os.path.join(output_debug,label+'_d.png')
                    # draw(label, boxes_d,  scores_d, infile, outfile, hot=False,color='blue')
                    #
                    # outfile = os.path.join(output_debug,label+'_th.png')
                    # draw(label, boxes_th, scores_th, infile, outfile, hot=False,color='red')

                    if(ground_truth is not None):
                        outfile = os.path.join(output_debug,label+'_gt.png')
                        val = ground_truth[label]
                        if(len(val)>0):
                            boxes_gt = [[v[0]-v[2]/2,v[1]-v[3]/2, v[0]+v[2]/2, v[1]+v[3]/2] for v in val]
                            scores_gt = [1.0 for v in val]
                            draw(label, boxes_gt, scores_gt, infile, outfile, hot=False, color='brown')

            with open(outputfile, 'wb') as f:
                pickle.dump(objects_dict, f, 2)

            if(ground_truth is not None):
                AP = {}
    
                for iou in [25, 50, 75]:
                    AP[iou] = eval_ap(objects_dict, ground_truth, iou / 100.)

                with open(outputfile.replace('.pkl', '.txt'), 'w') as f:
                    for iou in [25, 50, 75]:
                        value = round(AP[iou], 6)
                        print('AP{}: {}\n'.format(iou, value))
                        f.write('AP{}: {}\n'.format(iou, value))





if __name__ == "__main__":

    hot = False
    test = True
    # depth_modelpicklename = 'D:\\Testy\\Red-Hot-Deep-Blue\\dgx\\depth-faster\\training\\run_2020-02-02-210627_model_best.pkl'
    # thermal_modelpicklename = 'D:\\Testy\\Red-Hot-Deep-Blue\\dgx\\thermal-faster\\training\\run_2020-02-02-135141_model_best.pkl'

    if(test):
        strpart = 'test'
    else:
        strpart = 'valid'

    draw_debug = False

    strfix = '-faster'

    # imdir_thermal = 'D:\\Dane\\PrivPres\\iphd_'+strpart+'_thermal-v2\\images'
    # imdir_depth = 'D:\\Dane\\PrivPres\\iphd_' + strpart + '_depth\\images'
    # outputfile = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\input_fusion_'+strpart+'part\\res\\'+outputname
    # output_debug = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\input_fusion_'+strpart+'part\\res\\debug\\'

    imdir_thermal = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_' + strpart + '_thermal-v2/images'
    imdir_depth = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_' + strpart + '_depth/images'

    ref_path = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_validpart/ref'

    models_path = '/home/weronika/Dropbox/temp/logs/best'
    depth_modelpicklename = os.path.join(models_path,'depth',strpart,'run_2020-02-02-210627_model_best.pkl')
    thermal_modelpicklename = os.path.join(models_path,'thermal',strpart,'run_2020-01-30-151242_model_best.pkl')

    if(not(test)):
        with open(os.path.join(ref_path, 'ground_truth.pkl'), 'rb') as f:
            ground_truth = pickle.load(f)
    else:
        ground_truth = None

    for opt in ['my']: #['vanilla','none']:#,
        outputname = opt+ datetime.today().strftime('_%Y-%m-%d-%H%M%S')  +'_predictions.pkl'   #
        outputfile = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_' + strpart + 'part/res/' + outputname
        output_debug = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_' + strpart + 'part/res/debug/'

        if not os.path.exists(output_debug):
            os.makedirs(output_debug)

        merging(depth_modelpicklename,thermal_modelpicklename,outputfile, imdir_depth, imdir_thermal, ground_truth, output_debug, opt, draw_debug=draw_debug)

