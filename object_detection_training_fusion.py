import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from datasets import load_datasets_fusion, IPHDDataset, get_transform, MaskType
from engine import train_one_epoch_with_dualsource,evaluate, save_one_epoch
import utils
from models import my_model

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



def object_detection_training(outputdir, imdir_test, labeldir_test, imdir_train=None, labeldir_train=None, listfile_train=None,
                              hot=True, listfile_val=None, dgx=False, one_channel=False, pretrained=True,
                              n_epochs=10, max_iterations=1000, mask=MaskType.NoMask, only_faster=True, early_fusion=True):

    logdir = os.path.join(outputdir, 'logs')
    tempdir = os.path.join(outputdir, 'training')

    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # ==================== DATASETS ===================================================================================

    dataset_train, dataset_val = load_datasets_fusion(imdir=imdir_train, labeldir=labeldir_train, listfile=listfile_train, hot=hot,
                                               listfile_val=listfile_val, one_channel=one_channel, mask=mask)

    bs = 2
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
        print('{} devices on board'.format(torch.cuda.device_count()))


    # ============= MODEL =============================================

    if(early_fusion):
        print('Early fusion')
        model = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=only_faster,
                         fusion=True)
        model.to(device)
    else:
        print('Late fusion TO DO')
        model1 = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=only_faster,
                          fusion=True)
        model1.to(device)
        model2 = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=only_faster,
                          fusion=True)
        model2.to(device)



    # ============= OPTIMIZER =====================================
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)


    subdir = mask.name+datetime.today().strftime('_run_%Y-%m-%d-%H%M%S')
    tbwriter = SummaryWriter(os.path.join(logdir, subdir))


    best_model_file = os.path.join(outputdir,"training",subdir + "_model_best.pkl")

    # best_ap50 = 0
    best_ap75 = 0

    for epoch in range(n_epochs):
        if(early_fusion):
            train_one_epoch_with_dualsource(epoch, model, data_loader_train, tbwriter, device, optimizer, print_freq=100, max_iterations=max_iterations)
        else:
            train_one_epoch_with_dualsource(epoch, model1, data_loader_train, tbwriter, device, optimizer,
                                            print_freq=100, max_iterations=max_iterations, second_model=model2)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        ap50, ap75, _ = evaluate(epoch, model, data_loader_val, tbwriter,  device=device, max_iterations=max_iterations)
        # break

        if(ap75>best_ap75):
            print("SAVING MODEL")
            best_ap75 = ap75
            torch.save(model.state_dict(), best_model_file)

    # ============= TEST ON BEST =====================================
    dataset_test = IPHDDataset(imdir_test,
                               labeldir_test,
                               one_channel=True, hot=True, transforms=get_transform(train=False),
                               fusion=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=2*bs, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn)
    test_outputfile = os.path.join(outputdir, "training",subdir+"_predictions_train.pkl")

    model = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=only_faster, fusion=True)
    model.load_state_dict(torch.load(best_model_file), strict=False)
    model.eval()
    model.to(device)
    save_one_epoch(model, test_data_loader, test_outputfile, device)

