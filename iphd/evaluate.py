import pickle
import sys
import os
import numpy as np
import errno
import warnings
from PIL import Image, ImageDraw
from matplotlib import cm


nb_decimals = 6

def xywh2xyxy(x):
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.
    y[..., 1] = x[..., 1] - x[..., 3] / 2.
    y[..., 2] = x[..., 0] + x[..., 2] / 2.
    y[..., 3] = x[..., 1] + x[..., 3] / 2.
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_batch_statistics(outputs, targets, iou_threshold, norm=False):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for i, (sample_i, output) in enumerate(outputs.items()):

        if len(output) == 0:
            continue

        output = np.array(output)

        pred_boxes = xywh2xyxy(output[:, :4])
        pred_scores = output[:, 4]
        pred_labels = np.zeros(pred_boxes.shape[0])  # only 1 label, so these are dummy

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[sample_i]
        target_labels = np.zeros(len(annotations)) if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = xywh2xyxy(np.array(annotations))

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                ret = bbox_iou(pred_box[np.newaxis,:], target_boxes)
                box_index = np.argmax(ret, axis=0)
                iou = ret[box_index]

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def eval_ap(outputs, targets, iou_threshold=0.5):

    sample_metrics = get_batch_statistics(outputs, targets, iou_threshold=iou_threshold, norm=True)
    if (len(sample_metrics) < 1):
        print("Warning: empty targets")
        return 0
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # count groundtruth annotations
    n_truth = np.sum([len(val) for key, val in targets.items()])
    # labels list are all human class
    truth_labels = np.zeros(n_truth)  # dummy variable to be able to use ap_per_class
    precision, recall, AP, f1, class_id = ap_per_class(true_positives, pred_scores, pred_labels, truth_labels)

    return AP[0]  # we know there is only one class (human)

def draw_rectangles(imdir,outdir,pred_dict, hot):
    for kvp in pred_dict.items():
        label = kvp[0]
        boxes = kvp[1]
        imname = label+'.png'
        infile = os.path.join(imdir,imname)
        outfile = os.path.join(outdir, imname)

        if os.path.isfile(infile):
            I = Image.open(infile)
            width, height = I.size

            DM = np.array(I)

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
            IM = Image.fromarray(DM)

            th = 0.5

            imd = ImageDraw.Draw(IM)

            for c, dets in enumerate(boxes):
                if(dets[4]>th):
                    recshape = [(dets[0] - dets[2] / 2) * width, (dets[1] - dets[3] / 2) * height,
                                (dets[0] + dets[2] / 2) * width, (dets[1] + dets[3] / 2) * height]
                    imd.text(((dets[0] - dets[2] / 2) * width, (dets[1] - dets[3] / 2) * height), "{:.2f}".format(dets[4]))

                    if (hot):
                        imd.rectangle(recshape, outline="green")  # , width=2)
                    else:
                        imd.rectangle(recshape, outline="red")  # , width=3)
            IM.save(outfile)




if __name__ == "__main__":
    input_path = sys.argv[1]
    ref_path = os.path.join(input_path, 'ref')
    res_path = os.path.join(input_path, 'res')

    filename = sys.argv[2]



    if(os.path.exists(os.path.join(ref_path, 'ground_truth.pkl'))):
        print("Evaluation with ground truth")
        output_path = os.path.join(input_path, 'scores')
        try:
            os.makedirs(output_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        with open(os.path.join(ref_path, 'ground_truth.pkl'), 'rb') as f:
            ground_truth = pickle.load(f)

        with open(os.path.join(res_path, filename), 'rb') as f:
            predictions = pickle.load(f)

        # --------------
        # Error handling
        # --------------

        # simplest check: same number of frames in both truth and predictions
        if len(ground_truth) != len(predictions):
            print("Evaluation error: number of frames in predictions do not match the number of groundtruth frames. \
                   Cannot perform evaluation.")
            quit()

        # more in-depth check: all groundtruth frames have corresponding entry in predictions.
        missing = 0
        for fid in ground_truth.keys():
            if not fid in predictions:
                warnings.warn(fid)
                missing += 1

        if missing:
            print("Evaluation error: {} groundtruth frames do not have corresponding predictions. Make sure all frames in \
                   validation/test are predicted. If one frame does not contain detections, provide empty Python list. \
                   Cannot perform evaluation.".format(missing))
            quit()

        # ------------------
        # Metric computation
        # ------------------

        AP = {}

        for iou in [25, 50, 75]:
            AP[iou] = eval_ap(predictions, ground_truth, iou/100.)

        with open(os.path.join(output_path, filename.replace('.pkl','.txt')), 'w') as f:
            for iou in [25, 50, 75]:
                value = round(AP[iou], nb_decimals)
                print(value)
                f.write('AP{}: {}\n'.format(iou, value))

        quit()
    else:
        print("Drawing evaluation")

        output_path = os.path.join(input_path, filename.replace('.pkl',''))
        try:
            os.makedirs(output_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise



        with open(os.path.join(res_path, filename), 'rb') as f:
            predictions = pickle.load(f)

        hot = ('thermal' in input_path)
        if (hot):
            imdir = 'D:\\Dane\\PrivPres\\iphd_test_thermal\\images'
        else:
            imdir = 'D:\\Dane\\PrivPres\\iphd_test_depth\\images'


        draw_rectangles(imdir,output_path,predictions,hot)
