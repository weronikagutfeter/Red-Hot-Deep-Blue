import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from datasets import load_datasets, IPHDDataset, get_transform, MaskType
from engine import train_one_epoch,evaluate, draw_one_epoch, save_one_epoch
import utils
from models import my_model

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



def object_detection_training(outputdir, imdir_test, labeldir_test, imdir_train=None, labeldir_train=None, listfile_train=None,
                              hot=False, listfile_val=None, dgx=False, one_channel = False, pretrained=True,
                              preprocessing = 'none',
                              n_epochs=10, max_iterations=1000, mask=MaskType.NoMask, only_faster=False, labelpart='labels'):

    logdir = os.path.join(outputdir, 'logs')
    tempdir = os.path.join(outputdir, 'training')

    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # ==================== DATASETS ===================================================================================

    dataset_train, dataset_val = load_datasets(imdir=imdir_train, labeldir=labeldir_train, listfile=listfile_train, hot=hot,
                                               listfile_val=listfile_val, one_channel=one_channel, mask=mask, labelpart=labelpart,
                                               preprocessing=preprocessing)

    bs = 1#2
    num_workers = 2
    if(dgx):
        bs = 12
        num_workers = 2

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=bs, shuffle=True, num_workers=num_workers,  collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)

    # sanity_check(data_loader_train)
    # sanity_check(data_loader_val)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print("Current device: {}".format(torch.cuda.current_device()))

    # ============= MODEL =============================================

    model = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=only_faster, hot=hot, preprocessing=preprocessing)
    model.to(device)

    # ============= OPTIMIZER =====================================
    params = [p for p in model.parameters() if p.requires_grad]
    lr=0.005
    i = preprocessing.find("lr=")
    if(i>=0):
        ei = preprocessing.find('-',i)
        if(ei<0):
            ei = len(preprocessing)
        lr = float(preprocessing[i+3:ei])


    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)


    subdir = preprocessing+datetime.today().strftime('_run_%Y-%m-%d-%H%M%S')
    tbwriter = SummaryWriter(os.path.join(logdir, subdir))
    best_model_file = os.path.join(outputdir,"training",subdir + "_model_best.pkl")

    test_dataset = IPHDDataset(imdir_test,
                               labeldir_test,
                               hot=hot,
                               one_channel=one_channel,
                               transforms=get_transform(train=False),
                               preprocessing=preprocessing)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2 * bs, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)
    test_outputfile = os.path.join(outputdir, "training", subdir + "_predictions_train.pkl")

    # best_ap50 = 0
    best_ap75 = 0

    if(pretrained):
        evaluate(0, model, test_data_loader, tbwriter, device=device, header_prefix='Test')

    for epoch in range(n_epochs):
        train_one_epoch(epoch, model, data_loader_train, tbwriter, device, optimizer, print_freq=100, max_iterations=max_iterations)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        ap50, ap75,_ = evaluate(epoch, model, data_loader_val, tbwriter, device=device, max_iterations=(None if dgx else max_iterations))
        # break

        if(ap75>best_ap75):
            print("SAVING MODEL")
            best_ap75 = ap75
            torch.save(model.state_dict(), best_model_file)

    # ============= TEST ON BEST =====================================

    model = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=only_faster, hot=hot, preprocessing=preprocessing)
    model.load_state_dict(torch.load(best_model_file), strict=False)
    model.eval()
    model.to(device)
    evaluate(epoch, model, test_data_loader, tbwriter, device=device,header_prefix='Test')
    # save_one_epoch(model, test_data_loader, test_outputfile, device)

