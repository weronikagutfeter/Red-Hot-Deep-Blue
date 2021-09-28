import argparse
from latefusion import merging


parser = argparse.ArgumentParser()
parser.add_argument("imagedir_thermal", type=str, help="Path to directory with thermal files")
parser.add_argument("imagedir_depth", type=str, help="Path to directory with depth files")
parser.add_argument("-mt","--modelthermal", type=str, default="../models/thermal.pkl", help="Path to file with thermal model's weights")
parser.add_argument("-md","--modeldepth", type=str, default="../models/depth.pkl", help="Path to file with depth model's weights")
parser.add_argument("-o","--outputfile", type=str, default="lf_predictions.pkl", help="Path to file for writing predictions")

def main(imagedir_thermal, imagedir_depth,thermal_modelfile, depth_modelfile, outputfile, ground_truth=None, draw_debug=False):

    merging(depth_modelfile, thermal_modelfile, outputfile, imagedir_depth, imagedir_thermal, ground_truth,
            '', opt='my', draw_debug=draw_debug)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.imagedir_thermal,args.imagedir_depth, args.modelthermal, args.modeldepth, args.outputfile)