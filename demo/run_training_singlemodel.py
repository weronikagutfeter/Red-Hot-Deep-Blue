
from object_detection_training import object_detection_training
import argparse
import os
from datasets import create_listfile_for_dirs


parser = argparse.ArgumentParser()
parser.add_argument("trainimagedir", type=str, help="Path to training directory")
parser.add_argument("trainlistfile", type=str, help="Path to test file with image names")
parser.add_argument("valimagedir", type=str, help="Path to validation directory")
parser.add_argument("vallistfile", type=str, help="Path to validation file with image names")
parser.add_argument("modeltype", type=str, help="Model type thermal|depth")
parser.add_argument("-o","--outputdir", type=str, default="results", help="Path to directory where to save model")



def main(trainimagedir, trainlistfile, valimagedir,  vallistfile, modeltype, outputdir):


    if (modeltype == 'thermal'):
        hot = True
    elif (modeltype == 'depth'):
        hot = False
    else:
        raise ValueError("Model type should be 'depth' or 'thermal'")

    vallabeldir = valimagedir.replace('images','labels')
    trainlabeldir = trainimagedir.replace('images', 'labels')

    vallistfile = valimagedir.replace('images', 'list.txt')
    trainlistfile = trainimagedir.replace('images', 'list.txt')

    out_listfile_train = os.path.join(outputdir, 'list-train.txt')
    out_listfile_val = os.path.join(outputdir, 'list-val.txt')
    if not os.path.exists(out_listfile_val):
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        create_listfile_for_dirs(trainimagedir, trainlistfile, valimagedir, vallistfile, out_listfile_train,
                                 out_listfile_val,0)

    object_detection_training(outputdir, valimagedir, vallabeldir,
                                          listfile_train=out_listfile_train, hot=hot,listfile_val=out_listfile_val,
                              n_epochs=10,dgx=False)



if __name__ == '__main__':
    args = parser.parse_args()


    main(args.trainimagedir, args.trainlistfile, args.valimagedir, args.vallistfile, args.modeltype,  args.outputdir)

