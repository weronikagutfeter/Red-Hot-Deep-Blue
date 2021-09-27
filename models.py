from torchvision.models.detection import MaskRCNN, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
import torch.nn as nn
import torch

import torchvision

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',

    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}


def my_model(basic=True, pretrained=True, progress=True, one_channel=False,
                            num_classes=2, only_faster=False,
             pretrained_backbone=True, fusion=False, hot=True, preprocessing='none',
             **kwargs):


    if(basic):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    else:
        if pretrained:
            pretrained_backbone = False
        print ("Backbone is resnet-50 "+("pretrained" if pretrained else
                                         'default initialization')+ " expects "+("1" if one_channel else "3")+"-channel input")
        backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)

        if(fusion):
            # image_mean = [0.485, 0.456]
            # image_std = [0.229, 0.224]
            image_mean = [0.553, 0.223, 0]
            image_std = [0.128, 0.172, 1.0]
        else:
            if('resnet' in preprocessing):
                image_mean = [0.485, 0.456, 0.406]
                image_std = [0.229, 0.224, 0.225]
            elif('imagemean' in preprocessing):
                if(hot):
                    image_mean = [0.374, 0.374, 0.374 ]
                    image_std = [0.279, 0.279, 0.279 ]
                else:
                    image_mean = [0.193, 0.193, 0.193]
                    image_std = [0.177, 0.177, 0.177]
            elif ('no0mean' in preprocessing):
                if(hot):
                    image_mean = [0.553, 0.553, 0.553]
                    image_std = [0.128, 0.128, 0.128]
                else:
                    image_mean = [0.223, 0.223, 0.223]
                    image_std = [0.172, 0.172, 0.172]
            else:
                image_mean = [0.5, 0.5, 0.5]
                image_std = [1, 1, 1]




        if(only_faster):
            model = FasterRCNN(backbone, num_classes, image_mean=image_mean, image_std=image_std, **kwargs)
        else:
            model = MaskRCNN(backbone, num_classes, image_mean=image_mean, image_std=image_std, **kwargs)

        # if(fusion):
        #     model.backbone.body.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if(one_channel):
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            if(only_faster):
                pretrained_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                                           progress=progress)
            else:
                pretrained_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                                  progress=progress)
            model_dict = model.state_dict()
            # if(fusion):
            #     test = model_dict['backbone.body.conv1.weight']
            #     model_dict['backbone.body.conv1.weight'] = test[:,:2,:,:]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.shape == model_dict[k].shape}
            print('Loading pretrained weights: source len {} we take {}'.format(len(model_dict),len(pretrained_dict)))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    return model

def franken_model(depthweigths=None, thermalweights=None, num_classes=2, only_faster=True, **kwargs):


    backbone = resnet_fpn_backbone('resnet50',pretrained=True)
    if (only_faster):
        model_depth = FasterRCNN(backbone, num_classes, **kwargs)
        model_thermal = FasterRCNN(backbone, num_classes, **kwargs)
    else:
        model_depth = MaskRCNN(backbone, num_classes, **kwargs)
        model_thermal = MaskRCNN(backbone, num_classes, **kwargs)

    if(depthweigths is not None):
        depth_pretrained_dict = torch.load(depthweigths)

        model_dict_depth = model_depth.state_dict()
        depth_pretrained_dict = {k: v for k, v in depth_pretrained_dict.items() if
                           k in model_dict_depth and v.shape == model_dict_depth[k].shape}
        print('Loading pretrained weights for: source len {} we take {}'.format(len(model_dict_depth), len(depth_pretrained_dict)))
        model_dict_depth.update(depth_pretrained_dict)
        model_depth.load_state_dict(model_dict_depth)

    if(thermalweights is not None):
        thermal_pretrained_dict = torch.load(thermalweights)

        model_dict_thermal = model_thermal.state_dict()
        thermal_pretrained_dict = {k: v for k, v in thermal_pretrained_dict.items() if
                                 k in model_dict_thermal and v.shape == model_dict_thermal[k].shape}
        print('Loading pretrained weights for: source len {} we take {}'.format(len(model_dict_thermal),
                                                                                len(thermal_pretrained_dict)))
        model_dict_thermal.update(thermal_pretrained_dict)
        model_thermal.load_state_dict(model_dict_thermal)
    
    return model_depth, model_thermal