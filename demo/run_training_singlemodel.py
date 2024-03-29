
from object_detection_training import object_detection_training
import argparse
import os
from datasets import create_listfile_for_dirs


parser = argparse.ArgumentParser()
parser.add_argument("traindatadir", type=str, help="Path to training directory with images, labels and imagelist")
parser.add_argument("valdatadir", type=str, help="Path to validation directory with images, labels and imagelist")
parser.add_argument("modeltype", type=str, help="Model type thermal|depth")
parser.add_argument("-o","--outputdir", type=str, default="results", help="Path to directory where to save model")



def main(traindatadir, valdatadir,  modeltype, outputdir):


    if (modeltype == 'thermal'):
        hot = True
    elif (modeltype == 'depth'):
        hot = False
    else:
        raise ValueError("Model type should be 'depth' or 'thermal'")

    valimagedir = os.path.join(valdatadir, 'images')
    trainimagedir = os.path.join(traindatadir, 'images')

    vallabeldir = os.path.join(valdatadir,'labels')
    vallistfile = os.path.join(valdatadir, 'list.txt')
    trainlistfile = os.path.join(traindatadir, 'list.txt')

    out_listfile_train = os.path.join(outputdir, 'list-train.txt')
    out_listfile_val = os.path.join(outputdir, 'list-val.txt')
    if not os.path.exists(out_listfile_val):
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        create_listfile_for_dirs(trainimagedir, trainlistfile, valimagedir, vallistfile, out_listfile_train,
                                 out_listfile_val,0)

    object_detection_training(outputdir, valimagedir, vallabeldir,
                              listfile_train=out_listfile_train, listfile_val=out_listfile_val,
                              max_iterations=1000, dgx=True, n_epochs=10)



if __name__ == '__main__':
    args = parser.parse_args()


    main(args.traindatadir, args.valdatadir,  args.modeltype,  args.outputdir)

