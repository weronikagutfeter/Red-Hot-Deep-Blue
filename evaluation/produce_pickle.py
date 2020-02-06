import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from datasets import load_datasets, IPHDDataset, get_transform, divide_listfile
from engine import train_one_epoch,evaluate, draw_one_epoch, save_one_epoch
import utils
from models import my_model


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def object_detection_testing(modelfile, outputfile, imdir_test, hot=False, dgx=False, one_channel = False, pretrained=True):


    bs = 2
    num_workers = 2
    if(dgx):
        bs = 12
        num_workers = 2


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ============= MODEL =============================================

    model = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=True)
    model.load_state_dict(torch.load(modelfile), strict=False)
    # model = torch.load(modelfile)
    model.eval()
    model.to(device)

    # ============= OPTIMIZER =====================================


    test_dataset = IPHDDataset(imdir_test,
                                     None,
                                     hot=hot,
                                     one_channel=one_channel,
                                     transforms=get_transform(train=False))

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)

    save_one_epoch(model, test_data_loader, outputfile, device, hot)



if __name__ == "__main__":

    hot = True
    test = True
    modelpicklename = 'run_2020-01-30-151242_model_best.pkl'
    outputpart = modelpicklename.replace('.pkl','_predictions_onfast.pkl')
    if(test):
        strpart = 'test'
    else:
        strpart = 'valid'
    if (hot):
        hotpart = 'thermal'
    else:
        hotpart = 'depth'

    strfix = '-with-masks'

    imdir = 'D:\\Dane\\PrivPres\\iphd_'+strpart+'_'+hotpart+('-v2' if (hot and not(test)) else '')+'\\images'
    modelfile = 'D:\\Testy\\Red-Hot-Deep-Blue\\dgx\\'+hotpart+strfix+'\\training\\'+modelpicklename
    outputfile = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\input_'+hotpart+'_'+strpart+'part\\res\\'+outputpart

    object_detection_testing(modelfile,outputfile,imdir,hot=hot)

