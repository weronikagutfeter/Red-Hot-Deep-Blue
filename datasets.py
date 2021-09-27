import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import transforms as my_transforms
from torchvision import transforms as pytransforms
import utils
from engine import draw_evaluation
from enum import Enum

exclusion_list = ['vid00161_YrRitw9Q']

class MaskType(Enum):
    NoMask = 0
    Naive = 1
    Hot = 2
    Deep = 3
    HotNarrow = 4

class IPHDDataset(Dataset):
    def __init__(self, imdir=None, labeldir=None, filelisting = None, file_with_list=None, one_channel=False, hot=True, transforms=None, mask=MaskType.NoMask, fusion=False, labelpart='labels', preprocessing='none'):
        # self.imdir = imdir
        # self.labeldir = labeldir
        self.transforms = transforms
        self.one_channel = one_channel
        self.hot = hot
        self.filelisting = filelisting
        self.file_with_list = file_with_list
        self.fusion=fusion
        self.mask=mask
        self.labelpart = labelpart
        self.imgs, self.labels = self.load_images_and_labels(imdir,labeldir, labelpart)
        self.preprocessing = preprocessing


    def load_images_and_labels(self, imdir, labeldir, labelpart):
        print("Loading files...")
        print("Images are "+('fusion' if self.fusion else ('red-hot' if self.hot else 'deep-blue')))
        assert (imdir is not None or (self.filelisting is not None) or (self.file_with_list is not None)), "Not enough input data to load images"
        if(self.filelisting is None):
            if (self.file_with_list is not None):
                self.filelisting = []
                imgs = []
                labels = []
                with open(self.file_with_list) as lf:
                    for line in lf:
                        temp = line.strip()
                        if (len(temp) > 0):
                            labelfile = temp.replace('.png', '.txt').replace("images",labelpart)
                            if(self.check_labelfile(labelfile)):
                                self.filelisting.append(temp)
                                imgs.append(temp)
                                labels.append(labelfile)
            else:
                imgs = [os.path.join(imdir,elem) for elem in list(sorted(os.listdir(imdir)))]
                labels = []
                if(labeldir is not None):
                    labels = [os.path.join(labeldir,elem) for elem in list(sorted(os.listdir(labeldir)))]
        else:
            imgs = []
            labels = []
            for c, line in enumerate(self.filelisting):
                labelfile = line.replace('.png', '.txt').replace('images', labelpart)
                if(self.check_labelfile(labelfile)):
                    imgs.append(line)
                    labels.append(labelfile)
        return imgs, labels

    def check_labelfile(self,labelfilename):

        return (os.stat(labelfilename).st_size>0)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        if(self.fusion):
            img = img.resize((640,360),Image.ANTIALIAS)#
        img = np.array(img, dtype=np.double)
        if(self.fusion):
            height, width = img.shape
            # img = np.tile(img[:, :, np.newaxis], (1, 1, 2))
            if(self.hot):
                second_img_path = img_path.replace('thermal','depth').replace('-v2','')
                second_img = Image.open(second_img_path)
                second_img = second_img.resize((width,height),Image.ANTIALIAS)
                second_img = np.array(second_img, dtype=np.double)

                max_depth = 12000  # było 50
                second_img[second_img > max_depth] = max_depth
                second_img = (second_img / max_depth)

                img[img < 28500] = 28500  # clip at 283.15 K (or 10 ºC)
                img[img > 31500] = 31500  # clip at 313.15 K (or 40 ºC)
                img = ((img - 28500.) / (31500 - 28500))
            else:
                second_img_path = img_path.replace('depth','thermal')
                second_img = Image.open(second_img_path)
                second_img = second_img.resize((width, height), Image.ANTIALIAS)
                second_img = np.array(second_img, dtype=np.double)

                max_depth = 50000
                img[img > max_depth] = max_depth
                img = (img / np.max(img))

                second_img[second_img < 28315] = 28315
                second_img[second_img > 31315] = 31315
                second_img = ((second_img - 28315.) / (31315 - 28315))
            img = np.concatenate((img[:,:,np.newaxis],second_img[:,:,np.newaxis],np.zeros((height,width,1))),axis=2)
            img = np.uint8(img * 255)

        else:
            if(not(self.one_channel)):
                img = np.tile(img[:,:,np.newaxis],(1,1,3))
                height,width, channels = img.shape
            else:
                height, width = img.shape

            if not (self.hot):
                if('standard' in self.preprocessing):
                    max_depth = 12000 #było 50
                    img[img > max_depth] = max_depth
                    img = (img / max_depth)
                else:
                    img = (img/np.max(img))

            else:

                if('standard' in self.preprocessing):
                # img[img< 28315] = 28315  # clip at 283.15 K (or 10 ºC)
                # img[img > 31315] = 31315  # clip at 313.15 K (or 40 ºC)
                # img = ((img - 28315.) / (31315 - 28315))

                    img[img < 28500] = 28500  # clip at 283.15 K (or 10 ºC)
                    img[img > 31500] = 31500  # clip at 313.15 K (or 40 ºC)
                    img = ((img - 28500.) / (31500 - 28500))
                else:
                    img = (img - np.min(img))/ (np.max(img)-np.min(img))

            img = np.uint8(img * 255)

        boxes = []
        if(len(self.labels)>0):
            label_path = self.labels[idx]
            with open(label_path) as lf:
                for c, line in enumerate(lf):
                    dets = np.fromstring(line, dtype=float, sep=' ')
                    xmin = int((dets[1] - dets[3] / 2) * width)
                    xmax = int((dets[1] + dets[3] / 2) * width)
                    ymin = int((dets[2] - dets[4] / 2) * height)
                    ymax = int((dets[2] + dets[4] / 2) * height)
                    if((xmax-xmin)>5 and (ymax-ymin)>5):
                        boxes.append([xmin, ymin, xmax, ymax])
        num_objs = len(boxes)

        image_id = torch.tensor([idx])

        if num_objs > 0:

            if(self.mask==MaskType.Hot):
                masks = np.zeros((num_objs, height, width), dtype=bool)
                patch = ((img[:, :, 0] > (24. / 40 * 255)) & (img[:, :, 0] < (38./40*255)))
                for c in range(num_objs):
                    box = boxes[c]
                    masks[c,int(box[1]):int(box[3]),int(box[0]):int(box[2])]=patch[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            elif (self.mask == MaskType.HotNarrow):
                masks = np.zeros((num_objs, height, width), dtype=bool)
                patch = ((img[:, :, 0] > (30. / 40 * 255)) & (img[:, :, 0] < (38. / 40 * 255)))
                for c in range(num_objs):
                    box = boxes[c]
                    masks[c, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = patch[int(box[1]):int(box[3]),
                                                                                     int(box[0]):int(box[2])]

            elif(self.mask==MaskType.Naive):
                masks = np.zeros((num_objs, height, width),dtype=bool)
                for c in range(num_objs):
                    box = boxes[c]
                    masks[c,int(box[1]):int(box[3]),int(box[0]):int(box[2])]=True

            else:
                masks = np.zeros((num_objs, height, width), dtype=bool)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            #dummy background mask only for tests
            masks = np.ones((1, height, width))
            boxes = torch.as_tensor([[0, 0, width,height]], dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.int64)
            area = torch.as_tensor([width*height])
            iscrowd = torch.zeros((1,), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, trans_target = self.transforms(img, target)

        return img, img_path, trans_target

    def __len__(self):
        return len(self.imgs)



def create_listfile_for_dirs(imdir_train, listfile_train, imdir_val,  listfile_val, out_listfile_train, out_listfile_val, validation_to_training_extension_ratio):
    filelisting_train = []
    with open(listfile_train) as lf:
        for line in lf:
            temp = os.path.basename(line.strip())
            if (len(temp) > 0 and not(np.any([e in temp for e in exclusion_list]))):
                filelisting_train.append(os.path.join(imdir_train,temp))

    filelisting_val = []
    with open(listfile_val) as lf:
        for line in lf:
            temp = os.path.basename(line.strip())
            if (len(temp) > 0):
                filelisting_val.append(os.path.join(imdir_val,temp))


    print("{} train files {} validation files".format(len(filelisting_train), len(filelisting_val)))

    indices = torch.randperm(len(filelisting_val)).tolist()
    th_ind = int(np.ceil(0.1 * len(indices)))
    filelisting_val1 = [filelisting_val[i] for i in indices[-th_ind:]]
    filelisting_val2 = [filelisting_val[i] for i in indices[0:-th_ind]]
    filelisting_val = filelisting_val1
    if(validation_to_training_extension_ratio>100):
        validation_to_training_extension_ratio=1
        filelisting_train = []
    print("{} validation-validation files {} validation-training files to {} training-training files {} training in total".format(
        len(filelisting_val),
        validation_to_training_extension_ratio*len(filelisting_val2),
        len(filelisting_train),
        len(filelisting_train)+validation_to_training_extension_ratio*len(filelisting_val2)))
    for k in range(validation_to_training_extension_ratio):#4
        filelisting_train.extend(filelisting_val2)


    print("after random split: {} train files {} validation files".format(len(filelisting_train), len(filelisting_val)))


    with open(out_listfile_train, 'w') as f_train:
        for line in filelisting_train:
            f_train.write(line + '\n')

    with open(out_listfile_val, 'w') as f_val:
        for line in filelisting_val:
            f_val.write(line + '\n')

def create_listfile_for_dirs_simple(imdirs_train, imdir_val,  out_listfile_train, out_listfile_val):
    filelisting_train = []
    for imdir_train in imdirs_train:
        listfile_train = os.path.join(imdir_train.replace('images',''),'list.txt')
        with open(listfile_train) as lf:
            for line in lf:
                temp = os.path.basename(line.strip())
                if (len(temp) > 0 and not(np.any([e in temp for e in exclusion_list]))):
                    filelisting_train.append(os.path.join(imdir_train,temp))

    filelisting_val = []
    listfile_val = os.path.join(imdir_val.replace('images',''),'list.txt')
    with open(listfile_val) as lf:
        for line in lf:
            temp = os.path.basename(line.strip())
            if (len(temp) > 0):
                filelisting_val.append(os.path.join(imdir_val,temp))


    print("{} train files {} validation files".format(len(filelisting_train), len(filelisting_val)))

    with open(out_listfile_train, 'w') as f_train:
        for line in filelisting_train:
            f_train.write(line + '\n')

    with open(out_listfile_val, 'w') as f_val:
        for line in filelisting_val:
            f_val.write(line + '\n')


def divide_listfile(listfile, listfile_train, listfile_val):

    filelisting = []
    with open(listfile) as lf:
        for line in lf:
            temp = line.strip()
            if (len(temp) > 0):
                filelisting.append(temp)

    indices = torch.randperm(len(filelisting)).tolist()
    th_ind = int(np.ceil(0.02 * len(indices)))
    filelisting_val = [filelisting[i] for i in indices[-th_ind:]]
    filelisting_train = [filelisting[i] for i in indices[0:-th_ind]]

    print("{} train files {} validation files".format(len(filelisting_train),len(filelisting_val)))

    with open(listfile_train,'w') as f_train:
        for line in filelisting_train:
            f_train.write(line + '\n')

    with open(listfile_val,'w') as f_val:
        for line in filelisting_val:
            f_val.write(line + '\n')

def load_datasets(imdir=None,labeldir=None,listfile=None,hot=False, listfile_val=None, one_channel=False, mask=MaskType.NoMask, labelpart='labels', preprocessing='none'):
    assert ((labeldir is not None and imdir is not None) or (listfile is not None)), "Not enough input data to load images"

    if(listfile_val is None):
        filelisting = []
        with open(listfile) as lf:
            for line in lf:
                temp = line.strip()
                if(len(temp)>0):
                    filelisting.append(temp)

        indices = torch.randperm(len(filelisting)).tolist()
        th_ind = int(np.ceil(0.02*len(indices)))
        filelisting_val = [filelisting[i] for i in indices[-th_ind:]]
        filelisting_train = [filelisting[i] for i in indices[0:-th_ind]]

        dataset_train = IPHDDataset(imdir,
                                    labeldir,
                                    filelisting_train,
                                    one_channel=one_channel,
                                    hot=hot,
                                    transforms=get_transform(train=True),
                                    mask=mask,
                                    labelpart=labelpart,
                                    preprocessing=preprocessing)

        dataset_test = IPHDDataset(imdir,
                                   labeldir, filelisting_val,
                                    one_channel=one_channel, hot=hot, transforms=get_transform(train=False), mask=mask,
                                   preprocessing=preprocessing)
    else:
        #we have files with definitions
        dataset_train = IPHDDataset(file_with_list=listfile,
                              one_channel=one_channel,
                              hot=hot,
                              transforms=get_transform(train=True),mask=mask,
                                    labelpart=labelpart,
                                    preprocessing=preprocessing)

        dataset_test = IPHDDataset(file_with_list=listfile_val,
                                    one_channel=one_channel, hot=hot, transforms=get_transform(train=False),mask=mask,
                                   preprocessing=preprocessing)

    return dataset_train, dataset_test

def load_datasets_fusion(imdir=None,labeldir=None,listfile=None,hot=False, listfile_val=None, one_channel=False, mask=MaskType.NoMask,
                         preprocessing='none'):
    assert ((labeldir is not None and imdir is not None) or (listfile is not None)), "Not enough input data to load images"

    if(listfile_val is None):
        filelisting = []
        with open(listfile) as lf:
            for line in lf:
                temp = line.strip()
                if(len(temp)>0):
                    filelisting.append(temp)

        indices = torch.randperm(len(filelisting)).tolist()
        th_ind = int(np.ceil(0.02*len(indices)))
        filelisting_val = [filelisting[i] for i in indices[-th_ind:]]
        filelisting_train = [filelisting[i] for i in indices[0:-th_ind]]

        dataset_train = IPHDDataset(imdir,
                                    labeldir,
                                    filelisting_train,
                                    one_channel=one_channel,
                                    hot=hot,
                                    transforms=get_transform(train=True),
                                    mask=mask,
                                    fusion=True,
                                    preprocessing=preprocessing)

        dataset_test = IPHDDataset(imdir,
                                   labeldir, filelisting_val,
                                    one_channel=one_channel, hot=hot, transforms=get_transform(train=False), mask=mask,
                                   fusion=True,
                                    preprocessing=preprocessing)
    else:
        #we have files with definitions
        dataset_train = IPHDDataset(file_with_list=listfile,
                              one_channel=one_channel,
                              hot=hot,
                              transforms=get_transform(train=True),mask=mask,fusion=True,
                                    preprocessing=preprocessing)

        dataset_test = IPHDDataset(file_with_list=listfile_val,
                                    one_channel=one_channel, hot=hot, transforms=get_transform(train=False),mask=mask,fusion=True,
                                    preprocessing=preprocessing)

    return dataset_train, dataset_test



def get_transform(train):
    transforms = []
    transforms.append(my_transforms.ToTensor())
    if train:
        transforms.append(my_transforms.RandomHorizontalFlip(0.5))
        transforms.append(my_transforms.RandomCrop(0.5))
    return my_transforms.Compose(transforms)


def sanity_check(dataloader):

    sum_empty = 0
    sum_anomaly = 0
    sum_dummy = 0
    sum_outside = 0
    sum_full_outside = 0


    for b,(images,imagefilenames, targets) in enumerate(dataloader):

        empty = 0
        anomaly = 0
        dummy = 0
        outside = 0
        full_outside = 0


        for image, filename, target in zip(images,imagefilenames,targets):
            target_boxes = [t.cpu().numpy() for t in target['boxes']]
            labels = target['labels'].cpu().numpy()

            if(len(target_boxes)<1):
                empty += 1

            if(np.any(labels<1)):
                dummy+=1

            c,h,w = image.shape

            for t in target_boxes:
                if(t[2]<=t[0]):
                    anomaly +=1
                    print('{} [{} {} {} {}]'.format(filename,t[0],t[1],t[2],t[3]))
                    break
                if (t[3] <= t[1]):
                    anomaly += 1
                    print('{} [{} {} {} {}]'.format(filename,t[0],t[1],t[2],t[3]))
                    break
            for t in target_boxes:
                if (t[2] > w):
                    outside += 1
                    break
                if (t[0] < 0):
                    outside += 1
                    break
                if (t[3] > h):
                    outside += 1
                    break
                if (t[1] < 0):
                    outside += 1
                    break
            for t in target_boxes:
                if (t[0] > w):
                    full_outside += 1
                    break
                if (t[2] < 0):
                    full_outside += 1
                    break
                if (t[1] > h):
                    full_outside += 1
                    break
                if (t[3] < 0):
                    full_outside += 1
                    break



        sum_empty += empty
        sum_anomaly += anomaly
        sum_dummy += dummy
        sum_outside +=  outside
        sum_full_outside += full_outside

        print('\rBatch {}: empty target {}/{} anomaly boxes {}/{} dummy frames {}/{} outsiders {}/{}  FULL outsiders {}/{}'.format(b,empty,len(images),anomaly,len(images),dummy,len(images), outside, len(images),full_outside, len(images)), end='')
    print('To sum up: empty target {} anomaly boxes {} dummy frames {} outsiders {}   FULL outsiders {}'.format(sum_empty,  sum_anomaly, sum_dummy, sum_outside, sum_full_outside))

if __name__ == "__main__":
    hot = True
    dataset = IPHDDataset('/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_train_thermal-v2/images',
                          '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_train_thermal-v2/labels',
                          hot=hot,
                          mask=MaskType.NoMask,
                          transforms=get_transform(train=True))


    subset = torch.utils.data.Subset(dataset, [0,1,2,3])
    basedataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

    traindataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)


    for k,(image,imfilename, target) in enumerate(traindataloader):

        image = image[0].cpu()
        converter = pytransforms.ToPILImage()
        image = converter(image)
        target = target[0]
        imfilename = imfilename[0]
        boxes = [t.cpu().numpy() for t in target['boxes']]
        masks = [t.cpu().numpy() for t in target['masks']]


        outfilename = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/sandbox/{}.png'.format(k)
        draw_evaluation(outfilename, image, boxes=None, target_boxes=boxes, hot=hot,preprocess=False)

        # for m,mask in enumerate(masks):
        #     maskfilename = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/sandbox/{}_mask{}.png'.format(k,m)
        #     IM = Image.fromarray(np.uint8(mask)*255)
        #     print("max {} min {}".format(np.max(mask),np.min(mask)) )
        #     IM.save(maskfilename)
        if(k>20):
            break



