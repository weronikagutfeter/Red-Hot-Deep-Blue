import os
from glob import glob
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm
import pickle



if __name__ == '__main__':

    hot = True
    draw_masks = True

    if not(hot):
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_valid_depth'
        outputdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\input_depth_validationpart'
    else:
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_valid_thermal-v2'
        outputdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\input_thermal_validationpart'


    labellist = glob(os.path.join(sourcedir,'labels','*.txt'))
    objects_dict = dict()
    dummypred_dict = dict()
    for labelfile in labellist:
        labelname = os.path.splitext(os.path.basename(labelfile))[0]
        values = []
        dummy = []
        with open(labelfile) as lf:
            for c, line in enumerate(lf):
                dets = np.fromstring(line, dtype=float, sep=' ')
                values.append(dets[1:])
                dummydets = [dets[1], dets[2], 1.4*dets[3], 1.4*dets[4], c*0.1]
                dummy.append(dummydets)
        objects_dict[labelname]=values
        dummypred_dict[labelname] = dummy

    with open(os.path.join(os.path.join(outputdir,'ref'), 'ground_truth.pkl'), 'wb') as f:
        pickle.dump(objects_dict,f,-1)

    # with open(os.path.join(os.path.join(sourcedir,'res'), 'predictions.pkl'), 'wb') as f:
    #     pickle.dump(dummypred_dict,f,-1)