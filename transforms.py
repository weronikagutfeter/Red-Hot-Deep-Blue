import random
import torch
import numpy as np

from torchvision.transforms import functional as F



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, second_image=None):
        if(second_image is None):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image, target, second_image = t(image, target, second_image)
            return image, target, second_image


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target, second_image=None):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if(second_image is not None):
                second_image = second_image.flip(-1)
            bbox = target["boxes"]
            if(len(bbox)>0):
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        if(second_image is not None):
            return image, target, second_image
        else:
            return image, target

class RandomCrop(object):
    def __init__(self, prob, max_margin_height=0.2, max_margin_width=0.4):
        self.prob = prob
        self.max_margin_height = max_margin_height
        self.max_margin_width = max_margin_width

    def __call__(self, image, target, second_image=None):
        if random.random() < self.prob:
            height, width = image.shape[-2:]

            margin_width = np.int(self.max_margin_width*width)
            margin_height = np.int(self.max_margin_height * height)
            t_left = np.random.randint(0,margin_width)
            t_right = np.random.randint(-margin_width,0)
            t_top = np.random.randint(0,margin_height)
            t_bottom = np.random.randint(-margin_height,0)

            bbox = target["boxes"]
            new_bbox = []
            new_width = width -t_left +t_right
            new_height = height -t_top +t_bottom
            for box in bbox:
                dets = box + torch.Tensor([-t_left, -t_top, -t_left, -t_top])
                if (dets[0] >= new_width or dets[2] < 0 or dets[1] >= new_height or dets[3] < 0):
                    continue
                else:
                    dets[0] = max(dets[0],0)
                    dets[2] = min(dets[2],new_width)
                    dets[1] = max(dets[1],0)
                    dets[3] = min(dets[3],new_height)

                    if(dets[3]-dets[1]>10 and dets[2]-dets[0]>10):
                        new_bbox.append(dets)
            if(len(new_bbox)>0):
                target["boxes"] = torch.stack(new_bbox)
                image = image[:, t_top:t_bottom, t_left:t_right]
                if(second_image is not None):
                    second_image = second_image[:, t_top:t_bottom, t_left:t_right]

        if (second_image is not None):
            return image, target, second_image
        else:
            return image, target


class ToTensor(object):
    def __call__(self, image, target,second_image=None):
        image = F.to_tensor(image)
        if (second_image is not None):
            second_image = F.to_tensor(second_image)
            return image, target, second_image
        else:
            return image, target