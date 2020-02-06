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
                            num_classes=2, only_faster=False, pretrained_backbone=True, fusion=False, **kwargs):


    if(basic):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    else:
        if pretrained:
            pretrained_backbone = False
        print ("Backbone is resnet-50 "+("pretrained" if pretrained else
                                         'default initialization')+ " expects "+("1" if one_channel else "3")+"-channel input")
        backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
        if(only_faster):
            model = FasterRCNN(backbone, num_classes, **kwargs)
        else:
            model = MaskRCNN(backbone, num_classes, **kwargs)

        if(fusion):
            model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif(one_channel):
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            if(only_faster):
                pretrained_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                                           progress=progress)
            else:
                pretrained_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                                  progress=progress)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.shape == model_dict[k].shape}
            print('Loading pretrained weights: source len {} we take {}'.format(len(model_dict),len(pretrained_dict)))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    return model

def franken_model(depthweigths, thermalweights, num_classes=2, only_faster=True, **kwargs):


    backbone = resnet_fpn_backbone('resnet50',pretrained=True)
    if (only_faster):
        model_depth = FasterRCNN(backbone, num_classes, **kwargs)
        model_thermal = FasterRCNN(backbone, num_classes, **kwargs)
    else:
        model_depth = MaskRCNN(backbone, num_classes, **kwargs)
        model_thermal = MaskRCNN(backbone, num_classes, **kwargs)

    depth_pretrained_dict = torch.load(depthweigths)
    thermal_pretrained_dict = torch.load(thermalweights)

    model_dict_depth = model_depth.state_dict()
    depth_pretrained_dict = {k: v for k, v in depth_pretrained_dict.items() if
                       k in model_dict_depth and v.shape == model_dict_depth[k].shape}
    print('Loading pretrained weights for: source len {} we take {}'.format(len(model_dict_depth), len(depth_pretrained_dict)))
    model_dict_depth.update(depth_pretrained_dict)
    model_depth.load_state_dict(model_dict_depth)

    model_dict_thermal = model_thermal.state_dict()
    thermal_pretrained_dict = {k: v for k, v in thermal_pretrained_dict.items() if
                             k in model_dict_thermal and v.shape == model_dict_thermal[k].shape}
    print('Loading pretrained weights for: source len {} we take {}'.format(len(model_dict_thermal),
                                                                            len(thermal_pretrained_dict)))
    model_dict_thermal.update(thermal_pretrained_dict)
    model_thermal.load_state_dict(model_dict_thermal)
    
    return model_depth, model_thermal