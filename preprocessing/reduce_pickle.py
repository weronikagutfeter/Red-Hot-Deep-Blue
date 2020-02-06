import os
from glob import glob
import numpy as np
from PIL import Image
import pickle
import sys


if __name__ == '__main__':

    hot = False
    draw_masks = True

    pickle_path = sys.argv[1]
    output_path = sys.argv[2]


    with open(pickle_path, 'rb') as f:
        input_dict = pickle.load(f)
        output_dict = dict()

        for kvp in input_dict.items():
            label = kvp[0]
            values = kvp[1]

            new_values = list(filter(lambda v: v[4] > 0.5, values))
            output_dict[label]=new_values

        with open(output_path, 'wb') as f:
            pickle.dump(output_dict,f,-1)