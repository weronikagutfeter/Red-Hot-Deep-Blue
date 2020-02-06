import os
from datasets import create_listfile_for_dirs
from object_detection_training import object_detection_training



if __name__ == "__main__":
    for hot in [True,False]:
        if (hot):
            imdir_train = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_train_thermal-v2/images'
            imdir_val = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_thermal-v2/images'
            labeldir_val = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_thermal-v2/labels'
            listfile_train = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_train_thermal-v2/list.txt'
            listfile_val = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_thermal-v2/list.txt'
            outputdir = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/thermal'
        else:
            imdir_train = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_depth/images'
            imdir_val = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_depth/images'
            labeldir_val = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_depth/labels'
            listfile_train = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_depth/list.txt'
            listfile_val = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_valid_depth/list.txt'
            outputdir = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/depth'

        out_listfile_train = os.path.join(outputdir,'list-train.txt')
        out_listfile_val = os.path.join(outputdir,'list-val.txt')
        if not os.path.exists(out_listfile_val):
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            create_listfile_for_dirs(imdir_train,listfile_train,imdir_val,listfile_val,out_listfile_train, out_listfile_val)

        object_detection_training(outputdir, imdir_val, labeldir_val, listfile_train=out_listfile_train, hot=hot,listfile_val=out_listfile_val, n_epochs=10,dgx=True, use_masks=False)




