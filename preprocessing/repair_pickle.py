import os
from glob import glob
import numpy as np
from PIL import Image
import pickle



if __name__ == '__main__':

    hot = False
    draw_masks = True

    if not(hot):
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_valid_depth'
        outputdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\input_depth_validationpart'
    else:
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_valid_thermal-v2'
        outputdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\input_thermal_validationpart'


    with open(os.path.join(os.path.join(outputdir,'res'), 'my_predictions.pkl'), 'rb') as f:
        input_dict = pickle.load(f)
        output_dict = dict()

        imdir = os.path.join(sourcedir,'images')


        for kvp in input_dict.items():
            label = kvp[0]
            values = kvp[1]
            imname = os.path.join(imdir,label+'.png')
            I = Image.open(imname)
            w,h = I.size

            new_values = []
            for box in values:
                temp = np.array([(box[0]+box[2])/(2*w),(box[1]+box[3])/(2*h),(box[2]-box[0])/w,(box[3]-box[1])/h,box[4]])
                new_values.append(temp)
            output_dict[label]=new_values

        with open(os.path.join(os.path.join(outputdir,'res'), 'out_predictions.pkl'), 'wb') as f:
            pickle.dump(output_dict,f,-1)