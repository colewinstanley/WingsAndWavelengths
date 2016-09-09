''' ButterfliesByTemplate.py: handles templates
and detection of trays and butterflies in images from the
main project file'''

import os
import math

import cv2
import numpy as np
import mahotas as mh
from matplotlib import pyplot as plt

from skimage.morphology import skeletonize as skim_skeletonize

# from analysis import *       # analysis functions in separate file
from ProgressBar import progress, startProgress, endProgress
from analysis import regminmax, REG_MIN, REG_MAX

TEMP_WIDTH = 50
TEMPL_THRESHOLD = 0.53        # threshold for template matching
HASH_THRESHOLD = 31.5
BLUR_LEVEL = 6          # gaussian blur level on match image
FLAG_WIDTH = 38         # threshold for threshold shifting (length??)
WIDTH_FLAG_SHIFT = 7.5  # shift the threshold this much when the width < FLAG_WIDTH
MAX_AREA_OVERLAP = 0.25         # max area of overlap
SPECIES_DIFF = 0.30

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

# TODO need eventually to update the hash tables / have separate ones for different types (e.g. skippers)

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

def show(img):
    plt.imshow(img)
    # plt.show()

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
    # r, g, b = cv2.split(temp)
    # _, bool_ch = cv2.threshold()
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

def weighted_absolute_dist(truefalse, weighted):        
    weights = weighted - 0.5
    distArr = truefalse - weighted
    weightedDistArr = np.abs(weights * distArr)
    return np.sum(weightedDistArr)

def hamming(arr1, arr2, bool=True):
    # extended hamming for non-boolean arrays
    L = len(arr1)
    if L != len(arr2):
        raise ValueError("arr1 and arr2 in hamming must be of same length")
    if bool:
        return np.count_nonzero(arr1!=arr2) / float(L)
    else:
        return np.sum(np.abs(arr1 - arr2)) / float(L)

def hash_cluster(d, max_hamming=0.25, num_clusters=0, both=False):
    ''' hash_cluster: clusters 64-dimensional hash codes using hamming distances,
        which can be transformed to have Euclidean-like meaning.
        args:
          d: dict of index:hash to cluster
          max_hamming: float val of maximum hamming dist between centroids
          num_clusters: number of clusters to go to. (Overrides max_hamming if not 0,
          unless the both flag is true, in which case whichever is reached first
          will be used.)
    '''
    # TODO implement cluster density limit
    if num_clusters != 0:
        max_hamming = 1
    clusters = []
    for k, h in d.iteritems():
        clusters.append((set([k]), h))
    d = 0
    while (d < max_hamming) & (len(clusters) > num_clusters):
        d = 1.00
        ind1 = -1
        ind2 = -1
        for i, (s, h) in enumerate(clusters):
            for j in range(i+1, len(clusters)):
                ham_dist = hamming(h, clusters[j][1], bool=False)
                if ham_dist < d:
                    d = ham_dist
                    ind1 = i
                    ind2 = j
        try:
            L1 = len(clusters[ind1][0])
            L2 = len(clusters[ind2][0])
            h1 = clusters[ind1][1]
            h2 = clusters[ind2][1]
        except KeyError:
            print "nothing merged. ", len(clusters)
        else:
            new_ham = (h1*L1 + h2*L2)/float(L1+L2)
            clusters[ind1] = ((clusters[ind1][0] | clusters[ind2][0]), new_ham)
            del clusters[ind2]
    return clusters

def rangeoverlap(x1,x2,xr1,xr2):
    o = 0
    if (xr1 >= x1) and (xr2 <= x2): #rect is completely enclosed
        o = xr2 - xr1
    elif (x1 >= xr1) and (x2 <= xr2): #rect completely encloses
        o = x2 - x1
    elif x2 > xr2: #rect below
        o = xr2 - x1
    else:   #rect above
        o = x2 - xr1
    return o

def append_anti_alias(arr, rect, phash, dhash):      # grayimg for hashing
    Alias = False
    xr1,yr1,xr2,yr2,qr,fr,ph = rect
    w = xr2-xr1
    h = yr2-yr1
    dist_dhash = weighted_absolute_dist(dhash, dHash_table)
    dist_phash = weighted_absolute_dist(phash, pHash_table)
    hash_index = dist_phash*dist_dhash
    if hash_index > HASH_THRESHOLD:
        return
    qr = hash_index*qr**2
    for r in arr:
        x1,y1,x2,y2,q,_,_ = r
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
                    arr.append((xr1,yr1,xr2,yr2,qr,fr,ph))
    if not Alias:
        arr.append((xr1,yr1,xr2,yr2,qr,fr,ph))

# comb_for_aliases(arr) looks for and removes aliases in the passed array
 # based on the MAX_AREA_OVERLAP parameter at the top of this file. Always keeps
 # the rectangle of higher match quality out of a pair of overlapping rectangles.
 # works in place. Returns True if any aliases found and removed; otherwise False. 
def comb_for_aliases(arr):          #shuffle?   #replace append_anti_alias?
    arr.sort(key=lambda point: point[4]) #sort by descending quality
    any_found = False
    for i in range(0, len(arr)):    #prioritizes x2 high q
        try:                        
            x1,y1,x2,y2,_,_,_ = arr[i]
            a1 = (x2-x1)*(y2-y1)
        except IndexError:
            break       #outside arr range
        for j in range(i+1, len(arr)):          #probably would look better with enumerate()?
            try:
                xr1,yr1,xr2,yr2,_,_,_ = arr[j]
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
    '''helper function for detectTrays that takes a list of contours and identifies which
        ones have the correct dimensions, area, shape, etc to be trays, and labels the 
        trays by their size'''
    trays = []
    for c in conts:
        if cv2.contourArea(c) > 3500:
            rect = cv2.minAreaRect(c)
            angleFlag = False
            for p in range(-2,3):           # proabably unnecessarily large range
                if (90*p + 10) < rect[2] < (90*p + 80):
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

#  ===============================================================================================
# public methods for interface with main project file

def compare_phash(img1, img2):
    return hamming(pHash(img1), pHash(img2))

def find_species(s):
    ''' find_species: find the differentiated species in a tray using a hash clustering
        technique on the detected butterflies
        args:
          s: set of hashes to cluster
    '''
    clusters = hash_cluster({i:h for (i, h) in enumerate(s)}, max_hamming=SPECIES_DIFF)


def detectTrays(image_pickle):        # returns contour list of trays and tray lookup image
    ''''''
    image = np.loads(image_pickle)
    img = adaptive_3channel_thresh(image)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    toContours = []
    trays = []
    
    edges = cv2.Canny(image.copy(), 100, 200)
    edges_blur5 = cv2.GaussianBlur(edges.astype('float'), (0,0), sigmaX=4.15)
    
    edges_blur5_thr = cv2.adaptiveThreshold((edges_blur5).astype('uint8'), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 0)
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
            if filename != ".DS_Store":
                print "Invalid template file: " + filename
        else:
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
        width_flag = (t[2] < FLAG_WIDTH or t[3] < FLAG_WIDTH)
        results = cv2.matchTemplate(img.astype('uint8'), np.loads(t[0]), cv2.TM_SQDIFF_NORMED)
        resultsf = cv2.GaussianBlur(results, (0,0), sigmaX=BLUR_LEVEL)
        rmin = regminmax(resultsf, 9, REG_MIN)
        h, w = t[2], t[3]
        thr = (resultsf < (TEMPL_THRESHOLD - width_flag*WIDTH_FLAG_SHIFT))
        loc = np.where(rmin*thr)
        for pt in zip(*loc[::-1]):
            q = resultsf[pt[1]][pt[0]]
            x1, y1, x2, y2 = (pt[0], pt[1], pt[0]+w, pt[1]+h)
            cropped = image_unpickle[int(y1+h/10.):int(y2-h/10.),int(x1+w/10.):int(x2-w/10.)]
            ph = pHash(cropped)
            dh = dHash(cropped)
            append_anti_alias(butterflies, (x1, y1, x2, y2, q, t[4], ph), ph, dh) #format x1, y1, x2, y2, quality
        j += 1                      # TODO do both ph and dh?
        progress(50.*j/len(temps))
    
    #rotate (well actually flip across x=y) once and repeat
    img = np.array(zip(*img[::-1]))
    wimg = np.shape(img)[1]
    for t in temps:
        width_flag = (t[2] < FLAG_WIDTH or t[3] < FLAG_WIDTH)
        results = cv2.matchTemplate(img.astype('uint8'), np.loads(t[0]), cv2.TM_SQDIFF_NORMED)
        resultsf = cv2.GaussianBlur(results, (0,0), sigmaX=BLUR_LEVEL)
        rmin = regminmax(resultsf, 9, REG_MIN)
        h, w = t[2], t[3]
        thr = (resultsf < (TEMPL_THRESHOLD - width_flag*WIDTH_FLAG_SHIFT))
        loc = np.where(rmin*thr)
        for pt in zip(*loc[::-1]):      
            q = resultsf[pt[1]][pt[0]]
            x1, y1, x2, y2 = (pt[1], (wimg-1)-(pt[0]+w), pt[1]+h, (wimg-1)-(pt[0]))
            cropped = image_unpickle[int(y1+h/10.):int(y2-h/10.),int(x1+w/10.):int(x2-w/10.)]
            ph = pHash(np.array(zip(*cropped[::-1])))
            dh = dHash(np.array(zip(*cropped[::-1])))
            append_anti_alias(butterflies, (x1, y1, x2, y2, q, t[4], ph), ph, dh) #format x1, y1, x2, y2, quality
                                                                    #order of y switched from before because of flip and sorter
        j += 1
        progress(50.*j/len(temps))

    while comb_for_aliases(butterflies): #less memory intensive and more efficient to do with smaller arrays
        pass    # runs until no aliases left; not sure why I need to run that function more than once...
    
    butterflies.sort(key=lambda b: b[1])        # sort by y1 coordinate  
    clusters = hash_cluster({i:ph for i, (_,_,_,_,_,_,ph) in enumerate(butterflies)}, 
                            max_hamming=0.15)
    endProgress()
    return butterflies, clusters


