import argparse
import os
import pickle
from datetime import datetime
from run_latefusion import merging


parser = argparse.ArgumentParser()
parser.add_argument("imagedir", type=str, help="Path to directory with files")
parser.add_argument("modeltype", type=str, help="Model type thermal|depth")
parser.add_argument("modelfile", type=str, default="thermal.pkl", help="Path to file with model weights")
parser.add_argument("outputfile", type=str, default="res.txt", help="Path to file for writing predictions")

def main(imagedir_thermal, imagedir_depth,thermal_modelfile, depth_modelfile, outputfile, ground_truth=None, draw_debug=False):




    outputname = 'fusion_predictions.pkl'
    outputfile = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_' + strpart + 'part/res/' + outputname
    output_debug = '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Output/baseline/input_fusion_' + strpart + 'part/res/debug/'

    merging(depth_modelpicklename, thermal_modelpicklename, outputfile, imdir_depth, imdir_thermal, ground_truth,
            output_debug, opt='my', draw_debug=draw_debug)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.imagedir,args.modeltype.lower(), args,modelfile, args.outputfile)