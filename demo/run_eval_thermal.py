from evaluation.produce_pickle import object_detection_testing
import os


if __name__ == "__main__":

    hot = True
    test = False
    if(test):
        strpart = 'test'
    else:
        strpart = 'valid'
    if (hot):
        hotpart = 'thermal'
    else:
        hotpart = 'depth'

    strfix = '-with-masks'

    imdir =  '/media/weronika/Nowy/Dane/Red-Hot-Deep-Blue/Dane/iphd_'+strpart+'_'+hotpart+('-v2' if (hot and not(test)) else '')+'/images'
    modelfile = os.path.join('..','models','depth.pkl')
    outputfile = os.path.join('..','models','depth-'+strpart+'predictions.pkl')

    object_detection_testing(modelfile,outputfile,imdir,hot=hot)

