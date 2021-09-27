from evaluation.produce_pickle import object_detection_testing
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("imagedir", type=str, help="Path to directory with files")
parser.add_argument("modeltype", type=str, help="Model type thermal|depth")
parser.add_argument("-m", "--modelfile", type=str, default="../models/thermal.pkl", help="Path to file with model weights")
parser.add_argument("-o","--outputfile", type=str, default="predictions.pkl", help="Path to pickle file for writing predictions")

def main(imagedir,modeltype, modelfile, outputfile):

    if(modeltype=='thermal'):
        hot=True
    elif(modeltype=='depth'):
        hot=False
    else:
        raise ValueError("Model type should be 'depth' or 'thermal'")

    object_detection_testing(modelfile,outputfile,imagedir,hot=hot)

if __name__ == '__main__':
    args = parser.parse_args()

    main(args.imagedir,args.modeltype.lower(), args.modelfile, args.outputfile)