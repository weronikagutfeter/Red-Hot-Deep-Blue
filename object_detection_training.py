import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from datasets import load_datasets, IPHDDataset, get_transform, divide_listfile, sanity_check
from engine import train_one_epoch,evaluate, draw_one_epoch, save_one_epoch
import utils
from models import my_model

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def object_detection_training(outputdir, imdir_test, labeldir_test, imdir_train=None, labeldir_train=None, listfile_train=None,
                              hot=False, listfile_val=None, dgx=False, one_channel = False, pretrained=True,
                              n_epochs=10, max_iterations=1000, use_masks=True, only_faster=False):

    logdir = os.path.join(outputdir, 'logs')
    tempdir = os.path.join(outputdir, 'training')

    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # ==================== DATASETS ===================================================================================

    dataset_train, dataset_test = load_datasets(imdir=imdir_train, labeldir=labeldir_train, listfile=listfile_train, hot=hot, listfile_val=listfile_val, one_channel=one_channel, use_masks=use_masks)

    bs = 2
    num_workers = 2
    if(dgx):
        bs = 12
        num_workers = 2

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=bs, shuffle=True, num_workers=num_workers,  collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)

    # sanity_check(data_loader_train)
    # sanity_check(data_loader_test)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ============= MODEL =============================================
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=only_faster)
    model.to(device)

    # ============= OPTIMIZER =====================================
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)


    tbwriter = SummaryWriter(os.path.join(logdir, datetime.today().strftime('run_%Y-%m-%d-%H%M%S')))


    subdir = datetime.today().strftime('run_%Y-%m-%d-%H%M%S')
    model_file = os.path.join(outputdir,"training",subdir + "_model_last.pkl")
    best_model_file = os.path.join(outputdir,"training",subdir + "_model_best.pkl")

    best_ap50 = 0

    for epoch in range(n_epochs):
        train_one_epoch(epoch, model, data_loader_train, tbwriter, device, optimizer,print_freq=100, max_iterations=max_iterations)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        ap50 = evaluate(epoch, model, data_loader_test, tbwriter, tempdir, device=device, hot=hot,max_iterations=max_iterations)
        # break

        if(ap50>best_ap50):
            print("SAVING MODEL")
            best_ap50 = ap50
            torch.save(model.state_dict(), best_model_file)

        # torch.save(model.state_dict(), model_file)


    test_dataset = IPHDDataset(imdir_test,
                                     labeldir_test,
                                     hot=hot,
                                     one_channel=one_channel,
                                     transforms=get_transform(train=False))

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)
    test_outputfile = os.path.join(outputdir, "training",subdir+"_predictions_e{}_train.pkl".format(n_epochs))

    # torch.save(model, model_file)
    # save_one_epoch(model, test_data_loader, test_outputfile, device, hot)

