pfile = 'D:\\Testy\\Red-Hot-Deep-Blue\\baseline\\submit\\thermal\\27-01\\predictions.pkl'

import pickle

with open(pfile, 'rb') as fp:

    data = pickle.load(fp)

    c = 0
    for kvp in data.items():
        if(len(kvp[1])<1):
            c = c+1
    print("No boxes in {} images".format(c))

    print(len(data))