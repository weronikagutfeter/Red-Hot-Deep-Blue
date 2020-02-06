import os
from glob import glob
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm



if __name__ == '__main__':

    hot = False

    if not(hot):
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_train_depth'
        outdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\masks\\iphd_train_depth'
    else:
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_valid_thermal-v2'
        outdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\masks\\iphd_valid_thermal'



    if not os.path.exists(outdir):
        os.makedirs(outdir)

    imlist = glob(os.path.join(sourcedir,'images','*.png'))
    for imf in imlist:
        imname = os.path.basename(imf)
        labelfile = os.path.join(sourcedir,'labels',imname.replace('.png','.txt'))
        outfile = os.path.join(outdir,imname)

        if os.path.isfile(labelfile):
            I = Image.open(imf)
            width, height = I.size

            IM = np.array(I)



            if(hot):

                DM = np.zeros((height, width), dtype=np.uint8)
                DM[(IM > 30075) & (IM<31115)] = 1.0  # clip at 283.15 K (or 10 ºC)
                DM = np.uint8(DM * 255)

                Y = Image.fromarray(DM)
                imd = ImageDraw.Draw(Y)
                with open(labelfile) as lf:
                    for c, line in enumerate(lf):
                        dets = np.fromstring(line, dtype=float, sep=' ')
                        recshape = [(dets[1] - dets[3] / 2) * width, (dets[2] - dets[4] / 2) * height,
                                    (dets[1] + dets[3] / 2) * width, (dets[2] + dets[4] / 2) * height]

                        imd.rectangle(recshape, outline="green")  # , width=2)

                Y.save(outfile)
            else:
                with open(labelfile) as lf:
                    for c, line in enumerate(lf):
                        dets = np.fromstring(line, dtype=float, sep=' ')
                        recshape = [(dets[1] - dets[3] / 2) * width, (dets[2] - dets[4] / 2) * height,
                                    (dets[1] + dets[3] / 2) * width, (dets[2] + dets[4] / 2) * height]
                        irecshape = np.round(recshape).astype(np.int)
                        test = IM[irecshape[1]:irecshape[1]+irecshape[3],irecshape[0]:irecshape[0]+irecshape[2]]
                        test0 = test[(test>500) & (test<9500)]
                        # histo = np.histogram(test0,bins=50)
                        # ind = np.argmax(histo[0])
                        # medv = (histo[1][ind] + histo[1][ind+1])/2
                        medv = np.median(test0[:])
                        human_margin = 300
                        M = (IM>(medv-human_margin)) & (IM<(medv+human_margin))


                        DM = np.zeros((height, width), dtype=np.uint8)
                        DM[M]=1.0
                        DM = np.uint8(DM*255)
                        # DM = np.tile(DM[:, :, np.newaxis], (1, 1, 3))

                        Y = Image.fromarray(DM)
                        imd = ImageDraw.Draw(Y)

                        # imd.rectangle(recshape, outline="red")#, width=3)
                        out0file = outfile.replace('.png','-{}.png'.format(c))
                        Y.save(out0file)
        else:
            print(imname + " no label file")