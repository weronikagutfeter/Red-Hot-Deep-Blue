import os
from glob import glob
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm



if __name__ == '__main__':

    hot = False
    draw_masks = False

    if not(hot):
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_valid_depth'
        outdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\gt\\iphd_valid_depth'
    else:
        sourcedir = 'D:\\Dane\\PrivPres\\iphd_valid_thermal-v2'
        outdir = 'D:\\Testy\\Red-Hot-Deep-Blue\\gt\\iphd_valid_thermal'

    if(draw_masks):
        outdir = outdir+'_masks'

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

            if(draw_masks):
                DM = np.zeros((height,width), dtype=np.uint8)
            else:
                DM = np.array(I)

                if not (hot):
                    max_depth = 50000
                    DM[DM > max_depth] = max_depth
                    DM = (DM / np.max(DM))
                    DM = np.uint8(cm.bone(1-DM) * 255)
                else:
                    DM[DM < 28315] = 28315  # clip at 283.15 K (or 10 ºC)
                    DM[DM > 31315] = 31315  # clip at 313.15 K (or 40 ºC)
                    DM = ((DM - 28315.) / (31315 - 28315))
                    DM = np.uint8(cm.hot(DM) * 255)
            IM = Image.fromarray(DM)


            imd = ImageDraw.Draw(IM)
            with open(labelfile) as lf:
                for c, line in enumerate(lf):
                    dets = np.fromstring(line, dtype=float, sep=' ')
                    recshape = [(dets[1] - dets[3]/2)*width, (dets[2]-dets[4]/2)*height, (dets[1] + dets[3]/2)*width, (dets[2]+dets[4]/2)*height]
                    if(draw_masks):
                        imd.rectangle(recshape, outline=(c+1), fill=(c+1))

                    else:
                        if(hot):
                            imd.rectangle(recshape, outline="green")#, width=2)
                        else:
                            imd.rectangle(recshape, outline="red")#, width=3)
            IM.save(outfile)
        else:
            print(imname + " no label file")