# ButterfliesByTemplate.py: eventually, will handle templates 
# and detection of trays and butterflies in images from the 
# main project file

import os
import sys
import csv
import math
import time
import shutil
import datetime

import cv2
import numpy as np
import mahotas as mh
from matplotlib import pyplot as pylab      # should get rid of this
from matplotlib import pyplot as plt

from skimage.measure import compare_ssim
from skimage.morphology import skeletonize as skim_skeletonize

from analysis import *       # analysis functions in separate file
from ProgressBar import *
from image_pack import *

# from scipy.spatial.distance import hamming

# class color:    # Color characters for writing to the console
#     PURPLE = '\033[95m'
#     CYAN = '\033[96m'
#     DARKCYAN = '\033[36m'
#     BLUE = '\033[94m'
#     GREEN = '\033[92m'
#     YELLOW = '\033[93m'
#     RED = '\033[91m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
#     END = '\033[0m'

# class image_pack:
#     def __init__(self, vis, fluor, uv, near_ir, ir, w=0):
#         path_dict = {'vis':vis, 'fluor':fluor, 'uv':uv, 'near_ir':near_ir, 'ir':ir}      
#         self.color = {}
#         self.gray = {}                   
#         for key,path in path_dict.iteritems():
#             im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
#             if im is None:
#                 raise IOError('invalid image path passed')
#             im = np.rot90(im, 3)
#             if w != 0:
#                 dim = (w, int(im.shape[0] * (float(w) / im.shape[1])))
#                 im = cv2.resize(im, dim)
#             self.color[key] = im
#             self.gray[key] = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

# thresh_temp_folder = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/temps/thresh_temps'
# temp_folder = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/temps/working_template'
# vis_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0073.jpg'
# fluor_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0076.jpg'
# uv_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0077.jpg'
# near_ir_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0074.jpg'
# ir_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0075.jpg'
# results_folder = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/results'

TEMP_WIDTH = 50
TEMPL_THRESHOLD = 0.53        # threshold for template matching
HASH_THRESHOLD = 31.5
BLUR_LEVEL = 6          # gaussian blur level on match image
FLAG_WIDTH = 38         # threshold for threshold shifting (length??)
WIDTH_FLAG_SHIFT = 7.5  # shift the threshold this much when the width < FLAG_WIDTH
MAX_AREA_OVERLAP = 0.25         # max area of overlap

MIN_32ND_SMALLER = 85
MAX_32ND_SMALLER = 110
MIN_16TH_SMALLER = 189
MAX_16TH_SMALLER = 215
MIN_8TH_LARGER = 375
MAX_8TH_LARGER = 425
MIN_32ND_LARGER = 223
MAX_32ND_LARGER = 260
MIN_4TH_LARGER = 447
MAX_4TH_LARGER = 510

DRAW_ALL = -1

dHash_table = np.array([ 0.12,  0.,    0.12,  0.02,  0.18,  0.42,  0.4,   0.7,   0.02,  0.,    0.08,  0.04,
                         0.1,   0.3,   0.36,  0.96,  0.7,   0.96,  0.58,  0.44,  0.38,  0.66,  0.92,  0.98,
                         0.96,  0.72,  0.68,  0.3,   0.46,  0.86,  0.98,  0.92,  0.14,  0.16,  0.24,  0.5,
                         0.42,  0.16,  0.06,  0.12,  0.26,  0.18,  0.38,  0.66,  0.66,  0.2,   0.1,   0.,
                         0.98,  0.98,  0.92,  0.98,  0.76,  0.66,  0.36,  0.14,  0.94,  1.,    0.9,   0.92,
                         0.68,  0.62,  0.48,  0.5])

pHash_table = np.array([ 1.,    0.6,   1.,    0.14,  0.96,  0.56,  0.88,  0.24,  0.5,   0.46,  0.18,  0.6,
                         0.24,  0.38,  0.26,  0.5,   0.76,  1.,    0.,    0.28,  0.84,  0.16,  0.68,  0.08,
                         0.5,   0.3,   0.48,  0.6,   0.26,  0.5,   0.36,  0.5,   1.,    0.92,  0.62,  0.6,
                         0.04,  0.42,  0.02,  0.08,  0.48,  0.24,  0.54,  0.28,  0.46,  0.28,  0.4,   0.46,
                         0.9,   0.3,   0.14,  0.06,  0.76,  0.14,  0.4,   0.68,  0.4,   0.2,   0.42,  0.44,
                         0.4,   0.4,   0.48,  0.16])

#show(img)
#shows image (given as img) using pylab interface.
def show(img):
    pylab.imshow(img)
    pylab.show()

# def startProgress(title):
#     global progress_x
#     global st
#     st = time.time()
#     sys.stdout.write(title + ": [" + "-"*45 + "]" + chr(8)*46)
#     sys.stdout.flush()
#     progress_x = 0

# def progress(x):
#     global progress_x
#     x = int(x * 45 // 100)
#     sys.stdout.write("#" * (x - progress_x))
#     sys.stdout.flush()
#     progress_x = x

# def endProgress():
#     end = time.time()
#     sys.stdout.write("#" * (45 - progress_x) + "] time: %s sec\n" % str(end - st)[:5])
#     sys.stdout.flush()

def adaptive_3channel_thresh(img, window=611):
    channels = cv2.split(img)
    t = []
    for ch in channels:
        t.append(cv2.adaptiveThreshold(ch, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C 
            , cv2.THRESH_BINARY, window, -2))
    return t[0] & t[1] & t[2]

#create_thresh_temp(temp)
 #takes a pre-cropped template image in RGB/BGR in temp param
 #and performs an otsu threshold on each channel and creates a b/w
 #single-channel image that is black wherever ANY of the three channels
 #were below the threshold. Only white if none of the channels 
 #meet the threshold. Returns boolean image. If the save flag is used,
 #saves temp image as numpy arr to current directory using the name specified.
def create_thresh_temp(temp, save=None, folder=None):
    bool_ch = (temp > mh.thresholding.otsu(temp))
    r,g,b = cv2.split(bool_ch.astype('uint8'))
    l = r & g & b  #numpy magic 
    if save is not None:
        os.chdir(folder)
        np.save(save, l)
    return l

def grayscale_avg(img):
    r,g,b = cv2.split(img)
    return (r/3. + g/3. + b/3.).astype('uint8')

def pHash(img, resize=True):
    if len(img.shape) == 3:
        img = grayscale_avg(img)
    if resize:
        dim = (32, 32)
        img = cv2.resize(img, dim)
    dct = cv2.dct(img.astype('float'))[:8,:8]       # should try gaussian weights on frequencies
    mean = np.mean(dct[1:])
    return (dct > mean).flatten()

def dHash(img, resize=True, flatten=True):
    if len(img.shape) == 3:
        img = grayscale_avg(img)
    if resize:
        dim = (8, 8)
        img = cv2.resize(img, dim)
    grad = np.gradient(img.astype('float'))
    grad = grad[0] > 0
    if flatten:
        return grad.flatten()
    else:
        return grad

def weighted_absolute_dist(truefalse, weighted):        # this might be MSE in disguise?
    weights = weighted - 0.5
    distArr = truefalse - weighted
    weightedDistArr = np.abs(weights * distArr)
    return np.sum(weightedDistArr)

def rangeoverlap(x1,x2,xr1,xr2):
    o = 0
    if (xr1 >= x1) and (xr2 <= x2): #rect is completely enclosed
        o = xr2 - xr1
    elif (x1 >= xr1) and (x2 <= xr2): #rect completely encloses
        o = x2 - x1
    elif (x2 > xr2): #rect below
        o = xr2 - x1
    else:   #rect above
        o = x2 - xr1
    return o

def append_anti_alias(arr, rect, grayimg):      # grayimg for hashing
    Alias = False
    xr1,yr1,xr2,yr2,qr,fr = rect
    if rect[0] > rect[2]:
        xr1 = rect[2]
        xr2 = rect[0] # if the rectangle is upside down/
    if rect[1] > rect[3]:
        yr1 = rect[3]
        yr2 = rect[1] # backwards, correct
    w = xr2-xr1
    h = yr2-yr1
    cropped = grayimg[int(yr1+h/10.):int(yr2-h/10.),int(xr1+w/10.):int(xr2-w/10.)]
    dist_dhash = weighted_absolute_dist(dHash(cropped), dHash_table)
    dist_phash = weighted_absolute_dist(pHash(cropped), pHash_table)
    dist_dhash_b = weighted_absolute_dist(dHash(np.array(zip(*cropped[::-1]))), dHash_table) # for vertical-facing butterflies
    dist_phash_b = weighted_absolute_dist(pHash(np.array(zip(*cropped[::-1]))), pHash_table)
    hash_index = min(dist_phash*dist_dhash, dist_phash_b*dist_dhash_b)
    if hash_index > HASH_THRESHOLD:
        return
    qr = hash_index*qr**2
    for r in arr:
        x1,y1,x2,y2,q,f = r
        o = 0
        if (x1 <= xr2) and (xr1 <= x2) and (y1 <= yr2) and (yr1 <= y2): #there is x and y overlap
            ox = rangeoverlap(x1,x2,xr1,xr2)
            oy = rangeoverlap(y1,y2,yr1,yr2)
            ao = ox*oy
            a1 = (x2-x1)*(y2-y1)
            a2 = (xr2-xr1)*(yr2-yr1)
            if (ao > a1*MAX_AREA_OVERLAP) or (ao > a2*MAX_AREA_OVERLAP):
                Alias = True
                if q > qr:      #leaves original in if match is of equal or better quality (lower q)
                    arr.remove(r)
                    arr.append((xr1,yr1,xr2,yr2,qr,fr))
    if not Alias:
        arr.append((xr1,yr1,xr2,yr2,qr,fr))

# comb_for_aliases(arr) looks for and removes aliases in the passed array
 # based on the MAX_AREA_OVERLAP parameter at the top of this file. Always keeps
 # the rectangle of higher match quality out of a pair of overlapping rectangles.
 # works in place. Returns True if any aliases found and removed; otherwise False. 
def comb_for_aliases(arr):          #shuffle?   #replace append_anti_alias?
    arr.sort(key=lambda point: point[4]) #sort by descending quality
    any_found = False
    for i in range(0, len(arr)):    #prioritizes x2 high q
        try:                        
            x1,y1,x2,y2,q,f = arr[i]
            a1 = (x2-x1)*(y2-y1)
        except IndexError:
            break       #outside arr range
        for j in range(i+1, len(arr)):          #probably would look better with enumerate()?
            try:
                xr1,yr1,xr2,yr2,qr,fr = arr[j]
            except IndexError:
                break
            if (x1 <= xr2) and (xr1 <= x2) and (y1 <= yr2) and (yr1 <= y2):     # x and y overlap
                ox = rangeoverlap(x1,x2,xr1,xr2)
                oy = rangeoverlap(y1,y2,yr1,yr2)
                ao = ox * oy  #area of overlap
                a2 = (xr2-xr1)*(yr2-yr1)
                if ((ao > a1*MAX_AREA_OVERLAP) or (ao > a2*MAX_AREA_OVERLAP)):
                    arr.pop(j)
                    any_found = True
        if i >= len(arr):
            break
    return any_found

def findTrays(conts):
    trays = []
    for c in conts:
        if cv2.contourArea(c) > 3500:
            rect = cv2.minAreaRect(c)
            angleFlag = False
            for p in range(-2,3):           # proabably unnecessarily large range
                if (90*p + 16) < rect[2] < (90*p + 74):
                    angleFlag = True
            if angleFlag:
                continue
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)
            dist1 = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
            dist2 = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
            n = min(dist1, dist2)
            x = max(dist1, dist2)
            if MIN_32ND_SMALLER < n < MAX_32ND_SMALLER:
                if MIN_32ND_LARGER < x < MAX_32ND_LARGER:
                    trays.append((box, '1/32'))
            elif MIN_16TH_SMALLER < n < MAX_16TH_SMALLER:
                if MIN_32ND_LARGER < x < MAX_32ND_LARGER:
                    trays.append((box, '1/16'))
            elif MIN_32ND_LARGER < n < MAX_32ND_LARGER:
                if MIN_8TH_LARGER < x < MAX_8TH_LARGER:
                    trays.append((box, '1/8'))
            elif MIN_8TH_LARGER < n < MAX_8TH_LARGER:
                if MIN_4TH_LARGER < x < MAX_4TH_LARGER:
                    trays.append((box, '1/4'))
    return trays

def convert_temps(temp_path, target_path):
    os.chdir(target_path)
    for filename in os.listdir(temp_path):
        im = cv2.imread(temp_path + '/' + filename)
        if im is not None:
            create_thresh_temp(im, os.path.splitext(filename)[0])

def detectTrays(image_pickle):        # returns contour list of trays and tray lookup image
    ''' '''
    # print "here!"
    image = np.loads(image_pickle)
    img = adaptive_3channel_thresh(image)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    toContours = []
    trays = []
    
    edges = cv2.Canny(image.copy(), 100, 200)
    edges_blur5 = mh.gaussian_filter(edges.astype('uint8')*250, 4.15)
    
    edges_blur5_thr = cv2.adaptiveThreshold((edges_blur5*120).astype('uint8'), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 0)
    y,x = edges_blur5_thr.shape
    edges_blur5_thr[0:4] = 1
    edges_blur5_thr[y-4:y] = 1
    edges_blur5_thr[:,0:4] = 1
    edges_blur5_thr[:,x-4:x] = 1
    
    toContours.append(edges_blur5_thr)
    
    edges_dilate = cv2.dilate(edges, element, iterations=2).astype('bool')         # it=1?
    toContours.append(edges_dilate)
    dimg = img.copy()
    dimg2 = img.copy()
    dimg3 = img.copy()
    dimg = cv2.dilate(dimg, element, iterations=4)
    dimg2 = cv2.dilate(dimg2, element, iterations=6)
    dimg3 = cv2.dilate(dimg3, element, iterations=2)
    dimg = cv2.erode(dimg, element, iterations=3)
    dimg2 = cv2.erode(dimg2, element, iterations=3)
    dimg3 = cv2.erode(dimg3, element, iterations=3)
    
    toContours.append(dimg * edges_blur5_thr)
    toContours.append(dimg2 * edges_blur5_thr)
    toContours.append(dimg3 * edges_blur5_thr)
    toContours.append(edges_blur5_thr | edges_dilate)
    
    tray_areas = np.zeros(np.shape(edges_blur5_thr)).astype('uint8')
    
    for im in toContours:
        cont_list, _ = cv2.findContours(skim_skeletonize(im).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        trays2 = findTrays(cont_list)
        for tray in trays2:
            tray_area = np.zeros(np.shape(edges_blur5_thr))
            cv2.drawContours(tray_area, [tray[0]], DRAW_ALL, 1, cv2.cv.CV_FILLED)
            intersection = tray_area * tray_areas.astype('bool')
            if np.sum(intersection) < cv2.contourArea(tray[0]) / 4:
                cv2.drawContours(tray_areas, [tray[0]], DRAW_ALL, len(trays)+1, cv2.cv.CV_FILLED)   # creates a lookup image for which tray is where
                trays.append(tray)
    return trays, tray_areas

# start = time.time()

# print "retrieving images..."
# imgc = image_pack(vis_path, fluor_path, uv_path, near_ir_path, ir_path, 800)
# img_full = image_pack(vis_path, fluor_path, uv_path, near_ir_path, ir_path)  #w=4912

# img = adaptive_3channel_thresh(imgc.color['vis'])

# element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# toContours = []
# trays = []

# edges = cv2.Canny(imgc.color['vis'].copy(), 100, 200)
# edges_blur5 = mh.gaussian_filter(edges.astype('uint8')*250, 4.15)

# edges_blur5_thr = cv2.adaptiveThreshold((edges_blur5*120).astype('uint8'), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 0)
# y,x = edges_blur5_thr.shape
# edges_blur5_thr[0:4] = 1
# edges_blur5_thr[y-4:y] = 1
# edges_blur5_thr[:,0:4] = 1
# edges_blur5_thr[:,x-4:x] = 1

# toContours.append(edges_blur5_thr)

# edges_dilate = cv2.dilate(edges, element, iterations=2).astype('bool')         # it=1?
# toContours.append(edges_dilate)
# dimg = img.copy()
# dimg2 = img.copy()
# dimg3 = img.copy()
# dimg = cv2.dilate(dimg, element, iterations=4)
# dimg2 = cv2.dilate(dimg2, element, iterations=6)
# dimg3 = cv2.dilate(dimg3, element, iterations=2)
# dimg = cv2.erode(dimg, element, iterations=3)
# dimg2 = cv2.erode(dimg2, element, iterations=3)
# dimg3 = cv2.erode(dimg3, element, iterations=3)

# toContours.append(dimg * edges_blur5_thr)
# toContours.append(dimg2 * edges_blur5_thr)
# toContours.append(dimg3 * edges_blur5_thr)
# toContours.append(edges_blur5_thr | edges_dilate)

# tray_areas = np.zeros(np.shape(edges_blur5_thr)).astype('uint8')

# for im in toContours:
#     cont_list, _ = cv2.findContours(skim_skeletonize(im).astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     trays2 = findTrays(cont_list)
#     for tray in trays2:
#         tray_area = np.zeros(np.shape(edges_blur5_thr))
#         cv2.drawContours(tray_area, [tray[0]], DRAW_ALL, 1, cv2.cv.CV_FILLED)
#         intersection = tray_area * tray_areas.astype('bool')
#         if np.sum(intersection) < cv2.contourArea(tray[0]) / 4:
#             cv2.drawContours(tray_areas, [tray[0]], DRAW_ALL, len(trays)+1, cv2.cv.CV_FILLED)   # creates a lookup image for which tray is where
#             trays.append(tray)
def loadTemps(temp_folder, thresh_temp_folder):
    os.chdir(thresh_temp_folder)        #probably not needed
    for filename in os.listdir(thresh_temp_folder):         # shutil?
        os.remove(filename)
    for filename in os.listdir(temp_folder):
        im = cv2.imread(temp_folder + '/' + filename)
        if im is not None:
            create_thresh_temp(im, save=os.path.splitext(filename)[0], folder=thresh_temp_folder)

def getSizedTemps(thresh_temp_folder):
    temps = []
    #Note: thresholded temps must be stored in the correct directory as
    #numpy archives, i.e. .npy files if the LOAD_TEMPS flag is off
    
    for filename in os.listdir(thresh_temp_folder):
        try:
            temp = np.load(thresh_temp_folder + '/' + filename)
        except IOError:     #for .DS_Store
            temp = None
            if filename != ".DS_Store":
                print "Invalid template file: " + filename
        if temp is not None:
            dim = (TEMP_WIDTH, int(temp.shape[0] * (float(TEMP_WIDTH) / temp.shape[1])))
     #       temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp.astype('uint8'), dim).astype('bool')
            temps.append((temp, filename))
    
    temps_sized = []
    #temps_sized.append(temps[1], temps[2])
    
    for im in temps:
        for i in range(4, 50, 2):
            rat = 20. / (i)
            t = cv2.resize(im[0].astype('uint8'), (0,0), fx=rat, fy=rat)
            for k in range(4, 8, 1): # stretch (mostly stretch, not squish)
                index = len(temps_sized)
                ts = cv2.resize(t, (0,0), fx=(k/5.), fy=1)
                temps_sized.append((ts.dumps(), index, np.shape(ts)[0], np.shape(ts)[1], os.path.splitext(im[1])[0] + str(i) + '.' + str(k))) #k was im[1]
    return temps_sized

def detectButterflies(image_pickle, temps):
    startProgress("Detecting butterflies    ")
    image_unpickle = np.loads(image_pickle)
    img = adaptive_3channel_thresh(image_unpickle)
    j = 0
    butterflies = []
    for t in temps:
        if t[2] < FLAG_WIDTH or t[3] < FLAG_WIDTH:
            width_flag = True
        else:
            width_flag = False
        results = cv2.matchTemplate(img.astype('uint8'), np.loads(t[0]), cv2.TM_SQDIFF_NORMED)
        resultsf = mh.gaussian_filter(results, BLUR_LEVEL)
        rmin = mh.regmin(resultsf)
        h, w = t[2], t[3]           
        loc = np.where(rmin)
        for pt in zip(*loc[::-1]):
            q = resultsf[pt[1]][pt[0]]
            if q < (TEMPL_THRESHOLD - width_flag*WIDTH_FLAG_SHIFT):
                append_anti_alias(butterflies, (pt[0], pt[1], pt[0]+w, pt[1]+h, q, t[4]), image_unpickle) #format x1, y1, x2, y2, quality
        j += 1
        progress(50.*j/len(temps))
    
    #rotate (well actually flip across x=y) once and repeat
    img = np.array(zip(*img[::-1]))
    wimg = np.shape(img)[1]
    for t in temps:
        if t[2] < FLAG_WIDTH or t[3] < FLAG_WIDTH:
            width_flag = True
        else:
            width_flag = False
        results = cv2.matchTemplate(img.astype('uint8'), np.loads(t[0]), cv2.TM_SQDIFF_NORMED)
        resultsf = mh.gaussian_filter(results, BLUR_LEVEL)
        rmin = mh.regmin(resultsf)
        h, w = t[2], t[3]
        loc = np.where(rmin)
        for pt in zip(*loc[::-1]):      
            q = resultsf[pt[1]][pt[0]]
            if q < (TEMPL_THRESHOLD - width_flag*WIDTH_FLAG_SHIFT):
                append_anti_alias(butterflies, (pt[1], (wimg-1)-(pt[0]), pt[1]+h, (wimg-1)-(pt[0]+w), q, t[4]), image_unpickle) #format x1, y1, x2, y2, quality
                                                                        #order of y switched from before because of flip and sorter
        j += 1
        progress(50.*j/len(temps))

    while comb_for_aliases(butterflies): #less memory intensive and more efficient to do with smaller arrays
        pass    # runs until no aliases left; not sure why I need to run that function more than once...

    butterflies.sort(key=lambda b: b[1])
    endProgress()    
    # print "here3"
    return butterflies
#zip(*img)
#print "\nScan Completed."


# im_rects = imgc.color['vis'].copy()
# im_trays = imgc.color['vis'].copy()
# draw_rect_from_arr(im_rects, butterflies, td=tray_dict)
# draw_cont_from_arr(im_trays, trays)




        # eventually include drawer number in filename


# show(im_trays)
# show(edges_blur5_thr)
# for im in toContours:
#     show(im)



# end = time.time()
# print "total detection time: " + str(end - start) + " seconds"

#==============================================================================
#Begin image analysis
# 1. subtractive
# 2. edges = lower than and higher than in ir/uv
# start = time.time()

#img_full = image_pack(vis_path, fluor_path, uv_path, near_ir_path, ir_path)

# normalizes image against standard using average image values
# for subtractive analysis. Takes two gray images to work.
# currently uses standard deviation and mean

# def normalize_against(standard, image):     # eventually do this against the gray card or actual standard
#     rand = np.random.rand(*np.shape(image))     # noise introduced within each bit to smooth over 8-bit to 32-bit conversion
#     standard_mean, standard_dev = cv2.meanStdDev(standard)  # so histograms using this function are smooth, not comb-like
#     mean, dev = cv2.meanStdDev(image)
#     im = (image.astype('float32') + rand) - (mean + 0.5)      # 0.5 to account for mean of noise added
#     # print 1, im                             # works because mean(A union B) = (mean(A) + mean(B)) / 2 if size(A) = size(B)
#     im = im * (standard_dev / dev)
#     return im + standard_mean

# def norm_mean_only(standard, image):        # without standard deviation
#     std_mean = np.mean(standard)
#     mean = np.mean(image)
#     return image * (std_mean / mean)

# def subtractive(d, image_s):                 # returns dictionary of subtracted and normed images from param "image_s"
#     ret = {}
#     js = 0.
#     for key,im in d.iteritems():
#         sub = im - image_s
#         smin = np.amin(sub)
#         sub -= smin             #shift to 0 first
#         smax = np.amax(sub)
#      #   print smax
#         if key != 'vis':       #if image is not uniformly 0 (as in vis)
#             ret[key] = sub.astype('float')*(255./smax)
#         progress(50*(js/len(d)))
#         js += 1
#     return ret

# def coarse_contrast(img1, img2, unitt, dist, threshold, vis_lower):          # should probably consider using smaller transforms of input images
#     """compare coarse level contrast differences between two images.
#     returns image overlay of major differences.
#     Uses two gray images; must be of same shape. <- nope, takes two vis color images
#     img1 is the visible image (high-c)
#     checks only for higher contrast in img2
#     unitt is the boolean image of the bg of the butterfly/
#     used w red in fluor image"""
#     border = dist*2
#     vi = img1.copy()
#     fl = img2.copy()        # memory?
#     sqdist = int(math.sqrt(2) * dist / 2)                    # also could do half the directions and negate others as necessary (abs)
#     dirs = {"N": (0, -dist), "NE": (sqdist, -sqdist), "E": (dist, 0)
#             , "SE": (sqdist, sqdist), #"S": (0, dist), "SW": (-sqdist, sqdist)
#             #, "W": (-dist, 0), "NW": (-sqdist, -sqdist)
#             }                     # x, (-)y
#     grads1 = {}
#     grads2 = {}
#     for d, _ in dirs.iteritems():
#         grads1[d] = np.zeros(np.shape(img1))
#         grads2[d] = np.zeros(np.shape(img1))

#     shape = np.shape(img1)
#     b = np.zeros((shape[0]-(dist*2), shape[1]-(dist*2))).astype('bool')
#     subs = {}
#     img1_border = np.zeros((shape[0] + border*2, shape[1] + border*2))
#     img2_border = np.zeros((shape[0] + border*2, shape[1] + border*2))
#     img1_border[border:shape[0]+border, border:shape[1]+border] = img1
#     img2_border[border:shape[0]+border, border:shape[1]+border] = img2
#     vi = vi[dist:shape[0]-dist, dist:shape[1]-dist]
#     hi_vis_t = np.ones((shape[0]-(dist*2), shape[1]-(dist*2)))
#     for d, coor in dirs.iteritems():            # generates noise at edges...
#         grads1[d] = np.absolute(img1 - img1_border[border+coor[0]:shape[0]+border+coor[0], border+coor[1]:shape[1]+border+coor[1]])
#         grads2[d] = np.absolute(img2 - img2_border[border+coor[0]:shape[0]+border+coor[0], border+coor[1]:shape[1]+border+coor[1]])
#         grads1[d] = grads1[d][dist:shape[0]-dist, dist:shape[1]-dist]
#         grads2[d] = grads2[d][dist:shape[0]-dist, dist:shape[1]-dist]       # crop out border noise
#       #  normalize_against(grads1[d], grads2[d])
#       #  show(grads2[d])
#       #  show(grads1[d])
#         subs[d] = grads2[d] - grads1[d]
#         hi_vis = grads1[d] < vis_lower
#         hi_vis_t *= hi_vis
#     for d, coor in dirs.iteritems():
#         subs[d] *= hi_vis_t
#         loc = np.where(subs[d] > threshold)
#         redloc = zip(*np.where(unitt)[::-1])
#         for pt in zip(*loc[::-1]):
#             ept = (pt[0] + dirs[d][1], pt[1] + dirs[d][0])
#             if not unitt[pt[::-1]] and not unitt[ept[::-1]]:
#                 cv2.line(vi, pt, ept, 250) #subs[d][pt[::-1]])
#                 try:
#                     b[pt[::-1]] = True
#                     b[ept[::-1]] = True
#                 except IndexError:
#                     pass
#     # show(img2)
#     b250 = 250 - mh.gaussian_filter(b.astype('uint8')*250, dist/2.42).astype('uint8')

#     det_params = cv2.SimpleBlobDetector_Params()
#     det_params.minThreshold = 15
#     det_params.maxThreshold = 125
#     det_params.filterByArea = True
#     det_params.minArea = 15
#     det_params.filterByInertia = True
#     det_params.minInertiaRatio = 0.075
#     det_params.minDistBetweenBlobs = 10

#     detector = cv2.SimpleBlobDetector(det_params)
#     kpts = detector.detect(b250)

#     vi = cv2.drawKeypoints(vi, kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     bk = cv2.drawKeypoints(b250, kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     imgk = cv2.drawKeypoints(img1[dist:shape[0]-dist, dist:shape[1]-dist], kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     fl = fl[dist:shape[0]-dist, dist:shape[1]-dist]/2
#     flnk = cv2.drawKeypoints(fl.astype('uint8'), kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#     cv2.putText(imgk, "VISIBLE", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
#     cv2.putText(vi, "LINES", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
#     cv2.putText(bk, "BLURRED HITS", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
#     cv2.putText(flnk, "FLUORESCENT", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

#     rk = np.zeros(((shape[0]-(dist*2))*2, (shape[1]-(dist*2))*2, 3))
#     ts = (np.shape(rk)[0], np.shape(rk)[1], 3)
#     rk[0:ts[0]/2, 0:ts[1]/2] = imgk
#     rk[0:ts[0]/2, ts[1]/2:ts[1]] = bk
#     rk[ts[0]/2:ts[0], ts[1]/2:ts[1]] = vi
#     rk[ts[0]/2:ts[0], 0:ts[1]/2] = flnk

#     return kpts, rk, np.mean(b)*100

# r = 4912/800.
# startProgress("normalizing      (1 of 4)")
# normed = {}
# j = 0.
# for key,im in img_full.gray.iteritems():        #also adds the gray vis "normed" as itself
#     progress(100 * (j / len(img_full.gray)))
#     normed[key] = normalize_against(img_full.gray['vis'], im)
#     j += 1

# endProgress()
# r = 4912/800.
# os.chdir(results_folder)

# startProgress("subtractive      (2 of 4)")
# j = 0.
# subt = subtractive(normed, img_full.gray['vis'])
# for key,im in subt.iteritems():
#     progress(50 + (50 * (j / len(subt))))
#     color_map = cv2.applyColorMap(im.astype('uint8'), cv2.COLORMAP_JET)
#     for (i, (x1,y1,x2,y2,q,f)) in enumerate(butterflies):
#         os.chdir(crops_dirc + '/' + str(i+1))
#         cv2.imwrite(str(i+1) + "_" + key + '_sub.jpg', color_map[int(y1*r):int(y2*r), int(x1*r):int(x2*r)])
#     j += 1

# endProgress()

# gradientsx, gradientsy = {},{}      # look at x and y gradients (more organic way of looking at edges)
# for key,im in normed.iteritems():
#     gradientsy[key],gradientsx[key] = np.gradient(im)           # gets and unpacks grad(im); pixels are the multivariate gradient at that point.
#     #show(gradientsy[key])

# for (keyx,imx),(keyy,imy) in zip(subtractive(gradientsx, gradientsx['vis']).iteritems(), subtractive(gradientsy, gradientsy['vis']).iteritems()):
#     gradient_magnitude = cv2.magnitude(imx.astype('float'), imy.astype('float'))
#     color_map = cv2.applyColorMap(gradient_magnitude, cv2.COLORMAP_JET)
#     cv2.imwrite(keyx + '_grad-sub.jpg', color_map)


# TODO make dict of full sized crops


# startProgress("histograms       (3 of 4)")
# hists = list()
# for (i, (x1,y1,x2,y2,_,_)) in enumerate(butterflies):
#     progress(40 * (float(i) / len(butterflies)))
#     mask = np.zeros(img_full.gray['vis'].shape)
#     mask[int(y1*r):int(y2*r),int(x1*r):int(x2*r)] = 255
#     mask = mask.astype('uint8')
#     h = {'i':i}
#     for key,f in normed.iteritems():
#         h[key] = cv2.calcHist([f.astype('uint8')],[0],mask,[256],[0,256])
#     hists.append(h)



# j = 0.
# hists_compare_chisqr = {}
# hists_compare_corr = {}
# ssims = {}
# for d in hists:
#     progress(40 + (60 * (j / len(butterflies))))
#     h_i = str(d['i']+1)
#     os.chdir(crops_dirc + '/' + h_i)
#     comp_chisqr = {}
#     comp_corr = {}
#     ssim = {}
#     (x1,y1,x2,y2,_,_) = butterflies[d['i']]
#     d_list = list(enumerate(d.iteritems()))
#     max_level = max(hbin for _,(_,hist) in d_list if type(hist) is not int for hbin in hist)    # nested, filtered generator for ALL hist. \
#     for ind,(k1,hist1) in d_list:                                                               # bins in d_list across all the histograms
#         if k1 is not 'i':
#             pylab.plot(hist1)
#             pylab.xlim([0,256])
#             pylab.ylim([0, max_level+100])
#             pylab.savefig(k1 + "_hist" + ".jpg")
#             pylab.clf()
#             for ind2 in range(ind, len(d_list)):        # loop through the remaining histograms
#                 _,(k2,hist2) = d_list[ind2]
#                 if (k1 != k2) and (k2 is not 'i'):      # Don't think they'd be the same key but worth checking just in case
#                     comp_chisqr[k1 + ":" + k2] = cv2.compareHist(d[k1], d[k2], cv2.cv.CV_COMP_CHISQR)
#                     comp_corr[k1 + ":" + k2] = cv2.compareHist(d[k1], d[k2], cv2.cv.CV_COMP_CORREL)
#                     ssim[k1 + ":" + k2] = compare_ssim(normed[k1][int(y1*r):int(y2*r),int(x1*r):int(x2*r)]
#                                                       ,normed[k2][int(y1*r):int(y2*r),int(x1*r):int(x2*r)])
#     hists_compare_chisqr[d['i']] = comp_chisqr
#     hists_compare_corr[d['i']] = comp_corr
#     ssims[d['i']] = ssim
#     if comp_corr['ir:fluor'] > 0.45:
#         flags[d['i']]['no_var_sig'] = True
#     j += 1

# endProgress()



# keypoints = {}
# cont_totals = {}
# startProgress("contrast         (4 of 4)")
# flr, flg, flb = cv2.split(img_full.color['fluor'])
# vir, vig, vib = cv2.split(img_full.color['vis'])
# bg = (flb>200)|(vib>133)|(flr>70)
# for (i, (x1,y1,x2,y2,_,_)) in enumerate(butterflies):  # must be a more effecient / better looking way to do this
#     progress(100 * (float(i) / len(butterflies)))       # FIX ISSUES WITH OVERFLOW IN NORMED ARRS
#     vis = img_full.gray['vis'][int(y1*r):int(y2*r),int(x1*r):int(x2*r)]
#     fln = norm_mean_only(vis, img_full.gray['fluor'][int(y1*r):int(y2*r),int(x1*r):int(x2*r)])
#     bgc = bg[int(y1*r):int(y2*r),int(x1*r):int(x2*r)]
#     kpts, vi, cont_total = coarse_contrast(vis, fln*2, bgc, 12, 45, 64)
#     keypoints[i] = kpts
#     cont_totals[i] = cont_total
#     os.chdir(crops_dirc + '/' + str(i+1))
#     cv2.imwrite(str(i+1) + "_fl_cont" + '.jpg', vi)

# endProgress()

# output_report = 'report.csv'
# os.chdir(results_folder)
# with open(output_report, 'w') as outfile:               # create run report file
#     writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
#     writer.writerow(["time of run:", datetime.datetime.now()])
#     writer.writerow(["number of butterflies:", len(butterflies)])
#     writer.writerow([''])
#     writer.writerow(['key #', 'keypoint x', 'keypoint y', 'confidence', 'total contrast level'])
#     for i, _ in enumerate(butterflies):         # range??
#         for j, k in enumerate(keypoints[i]):
#             if j == 0:
#                 writer.writerow([i+1, int(k.pt[0]), int(k.pt[1]), str(k.size)[:5], cont_totals[i]])
#             else:
#                 writer.writerow(['', int(k.pt[0]), int(k.pt[1]), str(k.size)[:5]])
#     writer.writerow([''])
#     writer.writerow(['key #', 'histogram pair', 'chi-square', 'correlation', 'structural similarity', 'negative flag'])
#     for i,d in hists_compare_chisqr.iteritems():
#         for ii, (key, val) in enumerate(hists_compare_chisqr[i].iteritems()):
#             if ii == 0:
#                 writer.writerow([i+1, key, val, hists_compare_corr[i][key], ssims[i][key], flags[i]['no_var_sig']])
#             else:
#                 writer.writerow(['', key, val, hists_compare_corr[i][key], ssims[i][key]])

# end = time.time()
# print "total analysis time: " + str(end - start) + " seconds"

