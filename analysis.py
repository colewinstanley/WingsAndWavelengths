# analysis.py analysis functions to support ButterfliesByTemplate.py
import math
import numpy as np
import cv2
import time
from skimage.measure import compare_ssim
from scipy.ndimage.filters import minimum_filter
from ProgressBar import *

SAME_AS_SOURCE = -1
SOBEL_X = 0
SOBEL_Y = 1
REG_MIN = 0
REG_MAX = 1

def normalize_against(standard, image):     # eventually do this against the gray card or actual standard
    rand = np.random.rand(*np.shape(image))     # noise introduced within each bit to smooth over 8-bit to 32-bit conversion
    standard_mean, standard_dev = cv2.meanStdDev(standard)  # so histograms using this function are smooth, not comb-like
    mean, dev = cv2.meanStdDev(image)
    im = (image.astype('float32') + rand) - (mean + 0.5)      # 0.5 to account for mean of noise added
    # print 1, im                             # works because mean(A union B) = (mean(A) + mean(B)) / 2 if size(A) = size(B)
    im = im * (standard_dev / dev)
    return im + standard_mean

def norm_mean_only(standard, image):        # without standard deviation
    std_mean = np.mean(standard)
    mean = np.mean(image)
    return image * (std_mean / mean)

def subtractive(d, image_s):                 # returns dictionary of subtracted and normed images from param "image_s"
    ret = {}
    js = 0.
    for key,im in d.iteritems():
        sub = im - image_s
        smin = np.amin(sub)
        sub -= smin             #shift to 0 first
        smax = np.amax(sub)
     #   print smax
        if key != 'vis':       #if image is not uniformly 0 (as in vis)
            ret[key] = sub.astype('float')*(255./smax)
        progress(50*(js/len(d)))
        js += 1
    return ret

def coarse_contrast(img1, img2, unitt, dist, threshold, vis_lower):          # should probably consider using smaller transforms of input images
    """compare coarse level contrast differences between two images.
    returns image overlay of major differences.
    Uses two gray images; must be of same shape. <- nope, takes two vis color images
    img1 is the visible image (high-c)
    checks only for higher contrast in img2
    unitt is the boolean image of the bg of the butterfly/
    used w red in fluor image"""
    start = time.time()
    border = dist*2
    vi = img1.copy()
    fl = img2.copy()        # memory?
    sqdist = int(math.sqrt(2) * dist / 2)                    # also could do half the directions and negate others as necessary (abs)
    dirs = {"N": (0, -dist), "NE": (sqdist, -sqdist), "E": (dist, 0)
            , "SE": (sqdist, sqdist), #"S": (0, dist), "SW": (-sqdist, sqdist)
            #, "W": (-dist, 0), "NW": (-sqdist, -sqdist)
            }                     # x, (-)y
    grads1 = {}
    grads2 = {}
    for d, _ in dirs.iteritems():
        grads1[d] = np.zeros(np.shape(img1))
        grads2[d] = np.zeros(np.shape(img1))

    shape = np.shape(img1)
    b = np.zeros((shape[0]-(dist*2), shape[1]-(dist*2))).astype('bool')
    subs = {}
    img1_border = np.zeros((shape[0] + border*2, shape[1] + border*2))
    img2_border = np.zeros((shape[0] + border*2, shape[1] + border*2))
    img1_border[border:shape[0]+border, border:shape[1]+border] = img1
    img2_border[border:shape[0]+border, border:shape[1]+border] = img2
    vi = vi[dist:shape[0]-dist, dist:shape[1]-dist]
    hi_vis_t = np.ones((shape[0]-(dist*2), shape[1]-(dist*2))).astype('bool')
    for d, coor in dirs.iteritems():            # generates noise at edges...
        grads1[d] = np.absolute(img1 - img1_border[border+coor[0]:shape[0]+border+coor[0], border+coor[1]:shape[1]+border+coor[1]])
        grads2[d] = np.absolute(img2 - img2_border[border+coor[0]:shape[0]+border+coor[0], border+coor[1]:shape[1]+border+coor[1]])
        grads1[d] = grads1[d][dist:shape[0]-dist, dist:shape[1]-dist]
        grads2[d] = grads2[d][dist:shape[0]-dist, dist:shape[1]-dist]       # crop out border noise
      #  normalize_against(grads1[d], grads2[d])
      #  show(grads2[d])
      #  show(grads1[d])
        subs[d] = grads2[d] - grads1[d]
        hi_vis = grads1[d] < vis_lower
        hi_vis_t *= hi_vis

    print "1", time.time() - start
    start = time.time()
    for d, coor in dirs.iteritems():
        subs[d] *= hi_vis_t
        loc = np.where(subs[d] > threshold)
        redloc = zip(*np.where(unitt)[::-1])
        for pt in zip(*loc[::-1]):
            ept = (pt[0] + dirs[d][1], pt[1] + dirs[d][0])
            if not unitt[pt[::-1]] and not unitt[ept[::-1]]:
                cv2.line(vi, pt, ept, 250) #subs[d][pt[::-1]])
                try:
                    b[pt[::-1]] = True
                    b[ept[::-1]] = True
                except IndexError:
                    pass
    print "2", time.time() - start
    # show(img2)
    b250 = 250 - cv2.GaussianBlur(b.astype('uint8')*250, (0,0), sigmaX=dist/2.42).astype('uint8')

    det_params = cv2.SimpleBlobDetector_Params()
    det_params.minThreshold = 0
    det_params.maxThreshold = 125
    det_params.filterByArea = False
    det_params.minArea = 15
    det_params.filterByInertia = True
    det_params.minInertiaRatio = 0.075
    det_params.minDistBetweenBlobs = 55
    det_params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector(det_params)
    kpts = detector.detect(b250)

    vi = cv2.drawKeypoints(vi, kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    bk = cv2.drawKeypoints(b250, kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgk = cv2.drawKeypoints(img1[dist:shape[0]-dist, dist:shape[1]-dist], kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fl = fl[dist:shape[0]-dist, dist:shape[1]-dist]/2
    flnk = cv2.drawKeypoints(fl.astype('uint8'), kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(imgk, "VISIBLE", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(vi, "LINES", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(bk, "BLURRED HITS", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(flnk, "FLUORESCENT", (2,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    rk = np.zeros(((shape[0]-(dist*2))*2, (shape[1]-(dist*2))*2, 3))
    ts = (np.shape(rk)[0], np.shape(rk)[1], 3)
    rk[0:ts[0]/2, 0:ts[1]/2] = imgk
    rk[0:ts[0]/2, ts[1]/2:ts[1]] = bk
    rk[ts[0]/2:ts[0], ts[1]/2:ts[1]] = vi
    rk[ts[0]/2:ts[0], 0:ts[1]/2] = flnk

    return kpts, rk, np.mean(b)*100


def norm_pack(pack_full_gray):
	normed = {}
	j = 0.
	length = len(pack_full_gray)
	for key,im in pack_full_gray.iteritems():        #also adds the gray vis "normed" as itself
	    progress(100 * (j / length))
	    normed[key] = normalize_against(pack_full_gray['vis'], im)
	    j += 1
	return normed

def create_hist_dict(butterfly, normed_pack, index):
	r = 4912/800.
	(x1,y1,x2,y2,_,_) = butterfly
	mask = np.zeros(normed_pack['vis'].shape)
	mask[int(y1*r):int(y2*r),int(x1*r):int(x2*r)] = 255
	mask = mask.astype('uint8')
	h = {'i':index}
	for key,f in normed_pack.iteritems():
	    h[key] = cv2.calcHist([f.astype('uint8')],[0],mask,[256],[0,256])
	return h

def compare_hists(d, butterfly, normed):
    r = 4912/800.
    comp_chisqr = {}
    comp_corr = {}
    ssim = {}
    x1,y1,x2,y2,_,_ = butterfly
    d_list = list(enumerate(d.iteritems()))
    for ind,(k1,hist1) in d_list:
        if k1 is not 'i':
            for ind2 in range(ind, len(d_list)):
                _,(k2,hist2) = d_list[ind2]
                if (k1 != k2) and (k2 is not 'i'):
                    comp_chisqr[k1 + ":" + k2] = cv2.compareHist(d[k1], d[k2], cv2.cv.CV_COMP_CHISQR)
                    comp_corr[k1 + ":" + k2] = cv2.compareHist(d[k1], d[k2], cv2.cv.CV_COMP_CORREL)
                    ssim[k1 + ":" + k2] = compare_ssim(cv2.resize(
                    	normed[k1][int(y1*r):int(y2*r),int(x1*r):int(x2*r)], (0,0), fx=0.4, fy=0.4)
                                                      ,cv2.resize(
                        normed[k2][int(y1*r):int(y2*r),int(x1*r):int(x2*r)], (0,0), fx=0.4, fy=0.4))
    return d['i'], comp_chisqr, comp_corr, ssim         # (str, dict, dict, dict)

def regminmax(im, dim, min_max):
	''' regminmax(...): take image and find regional maxima, mostly based on overall image
		structure but also on the kernel size specified in dim param. This function uses 
		Sobel approximations for derivatives (defined in separate function) rather than gradients.
		args:
		  im (numpy.ndarray): grayscale image input as numpy.ndarray. must be uint8, 
		  	  int16, or uint16
		  dim (int): kernel size for derivative approximations. In theory, this only matters for 
		  	  image smoothing; that is, a noiseless image should produce identical results 
		  	  regardless of the kernel dimensions.
		  min_max (int): flag for finding either the minimum or maximum. Specified vals are REG_MIN or
		  	  REG_MAX.
		returns:
		  anonymous numpy.ndarray: boolean image of locations identified as local maxima/minima.
		  	  Use numpy.where to obtain list of coordinates.
	'''
	BETA = 256
	sobel_mag = cv2.magnitude(sobel(im, 5, SOBEL_X), sobel(im, 5, SOBEL_Y))
	hessian = hessian_matrix(im, 19, 9)
	sobel_mag = cv2.normalize(sobel_mag, alpha=0, beta=BETA, norm_type=cv2.NORM_MINMAX) # use range in boolean instead?
	# laplacian = cv2.normalize(laplacian, alpha=-BETA, beta=BETA, norm_type=cv2.NORM_MINMAX)
	peak_minima = minimum_filter(sobel_mag, 17)
	# where:
	# 1. pix is min in neighborhood, 2. pix sobel is > highest sobel/5, 3, concave correct
	return (peak_minima == sobel_mag) * (sobel_mag < BETA/5) * (hessian > 0.0025)
	

def laplacian_approx(im, ksize, min_max):		# targets -> positive
	kernel1D = cv2.getGaussianKernel(ksize, 4.5) - cv2.getGaussianKernel(ksize, 2.5)
	if min_max == REG_MAX:
		kernel1D = -kernel1D
	elif min_max != REG_MIN:
		raise ValueError("invalid REG_MIN/REG_MAX")
	print "l", kernel1D
	return cv2.sepFilter2D(im, cv2.CV_32F, kernel1D, kernel1D)

def sobel(im, ksize, axis):
	if axis == SOBEL_X:
		kernelX = np.array(range((1-ksize)/2, (ksize+1)/2))
		kernelY = cv2.getGaussianKernel(ksize, -1)
	elif axis == SOBEL_Y:
		kernelY = np.array(range((1-ksize)/2, (ksize+1)/2))
		kernelX = cv2.getGaussianKernel(ksize, -1)
	else:
		raise ValueError("invalid Sobel axis")
	return cv2.sepFilter2D(im, cv2.CV_32F, kernelX, kernelY)

def hessian_matrix(im, ksize_first, ksize_second):
	fx = sobel(im, ksize_first, SOBEL_X)
	fy = sobel(im, ksize_first, SOBEL_Y)
	fxx = sobel(fx, ksize_second, SOBEL_X)
	fyy = sobel(fy, ksize_second, SOBEL_Y)
	fxy = sobel(fx, ksize_second, SOBEL_Y)
	return fxx + fyy - 2*fxy



