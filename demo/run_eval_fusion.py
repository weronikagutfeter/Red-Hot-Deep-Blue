import os
import pickle
from datetime import datetime
from run_fusion import merging


if __name__ == "__main__":

    ontestdir = True

    if(ontestdir):
        strpart = 'test'
    else:
        strpart = 'valid'

    draw_debug = False

    strfix = '-faster'



    imdir_thermal = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_' + strpart + '_thermal-v2/images'
    imdir_depth = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_' + strpart + '_depth/images'

    ref_path = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_validpart/ref'

    depth_modelpicklename = os.path.join('..','models','depth.pkl')
    thermal_modelpicklename = os.path.join('..','models','thermal.pkl')

    if(not(ontestdir)):
        with open(os.path.join(ref_path, 'ground_truth.pkl'), 'rb') as f:
            ground_truth = pickle.load(f)
    else:
        ground_truth = None


    outputname = 'fusion_predictions.pkl'
    outputfile = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_' + strpart + 'part/res/' + outputname
    output_debug = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_' + strpart + 'part/res/debug/'

    if not os.path.exists(output_debug):
        os.makedirs(output_debug)

    merging(depth_modelpicklename,thermal_modelpicklename,outputfile, imdir_depth, imdir_thermal, ground_truth, output_debug, opt='my', draw_debug=draw_debug)

