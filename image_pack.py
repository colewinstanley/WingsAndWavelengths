# image_pack.py: handle packing of images from files and drawing functions

import cv2
import os
import numpy as np

DRAW_ALL = -1

class color:    # Color characters for writing to the console
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class image_pack(object):
    def __init__(self, vis, fluor, uv, near_ir, ir, w=0):
        path_dict = {'vis':vis, 'fluor':fluor, 'uv':uv, 'near_ir':near_ir, 'ir':ir}      
        self.color = {}
        self.gray = {}                   
        for key,path in path_dict.iteritems():
            im = cv2.imread(os.getcwd() + '/' + path)
            if im is None:
                raise IOError('invalid image path passed:' + os.getcwd() + '/' + path)
            im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            im = np.rot90(im, 3)
            if w != 0:
                dim = (w, int(im.shape[0] * (float(w) / im.shape[1])))
                im = cv2.resize(im, dim)
            self.color[key] = im
            self.gray[key] = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

def draw_rect_from_arr(img, arr, add_temp_name, add_index, color=(0,0,255), width=2, td=None):       # specimens 
    for (i, (x1,y1,x2,y2,q,f,_)) in enumerate(arr):
        cv2.rectangle(img, (x1,y1), (x2,y2), color, width)
        if add_temp_name is True:
            cv2.putText(img, f, (x2+2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if add_index is True:
            if td is None:
                cv2.putText(img, str(i+1), (x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            else: 
                cv2.putText(img, str(i+1) + "." + str(td[i]), (x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

def draw_cont_from_arr(img, arr, color=(0,255,0), width=2):                 # trays
    for (i, (cont, label)) in enumerate(arr):
        cv2.drawContours(img, [cont], DRAW_ALL, color, width)
        cv2.putText(img, "TRAY " + str(i+1)
                , (int(np.mean(zip(*cont)[0])) - 47, int(np.mean(zip(*cont)[1])) - 10)
                , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,190,0), width)
        cv2.putText(img, "(" + label + ")"
                , (int(np.mean(zip(*cont)[0])) - 47, int(np.mean(zip(*cont)[1])) + 10)
                , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,190,0), width)
