
import torch
import torch.utils.data
from datasets import IPHDDataset, get_transform
from engine import  save_one_epoch
import utils
from models import my_model


def object_detection_testing(modelfile, outputfile, imdir_test, hot=False, dgx=False, one_channel = False, pretrained=True):


    bs = 2
    num_workers = 2
    if(dgx):
        bs = 12
        num_workers = 2


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ============= MODEL =============================================

    model = my_model(basic=False, pretrained=pretrained, one_channel=one_channel, only_faster=True)
    model.load_state_dict(torch.load(modelfile,map_location=device), strict=False)
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

    save_one_epoch(model, test_data_loader, outputfile, device)

