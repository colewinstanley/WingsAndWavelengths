# WingsAndWavelengths.py: main script file for Wings and Wavelengths project
# this file will handle image pickup and multiprocessing

import shutil
import os
import time
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult


import cv2
import numpy as np

from ButterfliesByTemplate import *
from image_pack import *
from analysis import *
from ProgressBar import *

thresh_temp_folder = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/temps/thresh_temps'
temp_folder = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/temps/working_template'
vis_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0073.jpg'
fluor_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0076.jpg'
uv_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0077.jpg'
near_ir_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0074.jpg'
ir_path = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/test_images/73/DSC_0075.jpg'
results_folder = '/Users/colewinstanley/Desktop/ShDet_Temp_Sandbox/results'

LOAD_TEMPS = True
ADD_TEMP_NAME = True
ADD_INDEX = True

def subtract_save(key, im, crops_dirc, butterflies):
	# im = np.loads(im_p)
	r = im.shape[1]/800.
	color_map = cv2.applyColorMap(im.astype('uint8'), cv2.COLORMAP_JET)
	for (i, (x1,y1,x2,y2,q,f)) in enumerate(butterflies):
	    os.chdir(crops_dirc + '/' + str(i+1))
	    cv2.imwrite(str(i+1) + "_" + key + '_sub.jpg', color_map[int(y1*r):int(y2*r), int(x1*r):int(x2*r)])

def save_hist(d, crops_dirc):
    h_i = str(d['i']+1)
    os.chdir(crops_dirc + '/' + h_i)
    max_level = max(hbin for _,hist in d.iteritems() if type(hist) is not int for hbin in hist)    # nested, filtered generator for ALL hist. \
    for k, hist in d.iteritems():                                                               # bins in d_list across all the histograms
        if k is not 'i':
            plt.plot(hist)
            plt.xlim([0,256])
            plt.ylim([0, max_level+100])
            plt.savefig(k + "_hist" + ".jpg")
            plt.clf()

def compare_contrast_wrapper(butterfly, index, bg, pack_full_gray, crops_dirc):
    r = pack_full_gray['vis'].shape[1]/800.
    vis = pack_full_gray['vis'][int(y1*r):int(y2*r),int(x1*r):int(x2*r)]
    fln = norm_mean_only(vis, pack_full_gray['fluor'][int(y1*r):int(y2*r),int(x1*r):int(x2*r)])
    bgc = bg[int(y1*r):int(y2*r),int(x1*r):int(x2*r)]
    kpts, vi, cont_total = coarse_contrast(vis, fln*2, bgc, 12, 45, 64)
    os.chdir(crops_dirc + '/' + str(i+1))
    cv2.imwrite(str(i+1) + "_fl_cont" + '.jpg', vi)
    return index, kpts, cont_total

def main():
	start = time.time()
	print "retrieving images..."
	imgc = image_pack(vis_path, fluor_path, uv_path, near_ir_path, ir_path, 800)
	img_full = image_pack(vis_path, fluor_path, uv_path, near_ir_path, ir_path)  #w=4912
	print "Done."

	if LOAD_TEMPS:
		loadTemps(temp_folder, thresh_temp_folder)

	detectionPool = Pool(processes=3)
	# res_temps = detectionPool.apply_async(getSizedTemps, args=(thresh_temp_folder,))
	templates = getSizedTemps(thresh_temp_folder)
	res_but = detectionPool.apply_async(detectButterflies, args=(imgc.color['vis'].dumps(), templates))
	# res_tr = detectionPool.apply_async(detectTrays, args=(imgc.color['vis'].dumps(),))
	trays, tray_areas = detectTrays(imgc.color['vis'].dumps())
	# res_temps.wait()	# wait for the templates to come in before butterfly detection is launched
	# templates = res_temps.get()
	# butterflies = detectButterflies(imgc.color['vis'], templates)
	detectionPool.close()
	# print 4
	# print "res", res_tr.get()
	# trays = res_tr.get()[0]
	# tray_areas = np.loads(res_tr.get()[1])
	# print 5
	butterflies = res_but.get()
	detectionPool.join()
	print "detection done"

	tray_dict = {}
	for (i, (x1,y1,x2,y2,_,_)) in enumerate(butterflies):
	    tray_dict[i] = tray_areas[(y1+y2)//2,(x1+x2)//2]

	im_rects = imgc.color['vis'].copy()
	im_trays = imgc.color['vis'].copy()
	draw_rect_from_arr(im_rects, butterflies, ADD_TEMP_NAME, ADD_INDEX, td=tray_dict)
	draw_cont_from_arr(im_trays, trays)

	crops_dirc = results_folder + '/crops'
	if not os.path.exists(crops_dirc):
	    os.makedirs(crops_dirc)

	startProgress("saving cropped images    ")
	os.chdir(crops_dirc)
	for (i, (x1,y1,x2,y2,_,_)) in enumerate(butterflies):
	    progress(100 * float(i) / len(butterflies))
	    os.chdir(crops_dirc)
	    try:
	        os.mkdir(str(i+1))
	    except OSError:
	        shutil.rmtree(str(i+1))
	        os.mkdir(str(i+1))
	    os.chdir(crops_dirc + '/' + str(i+1))
	    for key,im in img_full.color.iteritems():
	        cv2.imwrite(str(i+1) + "_" + key + '.jpg', cv2.cvtColor(im[int(y1*(4912/800.)):int(y2*(4912/800.)), 
	                                                                   int(x1*(4912/800.)):int(x2*(4912/800.))],
                                                                   cv2.COLOR_BGR2RGB))
	os.chdir(crops_dirc)
	cv2.imwrite("specimen_key.jpg", cv2.cvtColor(im_rects, cv2.COLOR_BGR2RGB))
	cv2.imwrite("tray_key.jpg", cv2.cvtColor(im_trays, cv2.COLOR_BGR2RGB))
	endProgress()

	flag_list = ['no_var_sig', 'hi_vis_uv', 'hi_fluor', 'ssim']
	flags = {i:{flag: False for flag in flag_list} for i,_ in enumerate(butterflies)}                # 2D dictionary of flags

	startProgress("normalizing      (1 of 4)")
	normed = norm_pack(img_full.gray)
	endProgress()

	startProgress("subtractive      (2 of 4)")
	j = 0.
	subt = subtractive(normed, img_full.gray['vis'])
	# subtractivePool = Pool(processes=5)
	# subtractivePool.map_async(subtract_save, ((key,im.dumps(),crops_dirc) for key,im in subt.iteritems()))
	[subtract_save(key,im,crops_dirc, butterflies) for key,im in subt.iteritems()]
	endProgress()
	# subtractivePool.close()
	
	
	# histogramPool = Pool(processes=5)
	# hists = histogramPool.map(create_hist_dict, ((butterfly, normed) for butterfly in butterflies))
	# histogramPool.map_async(save_hist, ((d, crops_dirc) for d in hists))
	# hist_results = histogramPool.map(compare_hists, ((d, butterflies) for d in hists))
	hists = [create_hist_dict(butterfly, normed, i) for i,butterfly in enumerate(butterflies)]
	[save_hist(d, crops_dirc) for d in hists]
	hist_results = [compare_hists(d, butterflies, normed) for d in hists]
	hists_compare_chisqr = {t[0]:t[1] for t in hist_results}
	hists_compare_corr = {t[0]:t[2] for t in hist_results}
	ssims = {t[0]:t[3] for t in hist_results}
	for k,d in hists_compare_corr.iteritems():
		if d['ir:fluor'] > 0.45:
			flags[d['i']]['no_var_sig'] = True
	# histogramPool.close()
	# # subtractivePool.join()
	# histogramPool.join()

	# contrastPool = Pool(processes=12)
	startProgress("contrast         (4 of 4)")
	flr, flg, flb = cv2.split(img_full.color['fluor'])
	vir, vig, vib = cv2.split(img_full.color['vis'])
	bg = (flb>200)|(vib>133)|(flr>70)
	# cont_results = contrastPool.map(
	# 			compare_contrast_wrapper, ((butterfly, i, bg, img_full.gray, crops_dirc) 
	# 			for i,butterfly in enumerate(butterflies)))
	cont_results = [compare_contrast_wrapper(butterfly, i, bg, img_full.gray, crops_dirc)
				for i,butterfly in enumerate(butterflies)]
	keypoints = {t[0]:t[1] for t in cont_results}
	cont_totals = {t[0]:t[1] for t in cont_results}
	# contrastPool.close()
	# contrastPool.join()
	endProgress()


	output_report = 'report.csv'
	os.chdir(results_folder)
	with open(output_report, 'w') as outfile:               # create run report file
	    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
	    writer.writerow(["time of run:", datetime.datetime.now()])
	    writer.writerow(["number of butterflies:", len(butterflies)])
	    writer.writerow([''])
	    writer.writerow(['key #', 'keypoint x', 'keypoint y', 'confidence', 'total contrast level'])
	    for i, _ in enumerate(butterflies):         # range??
	        for j, k in enumerate(keypoints[i]):
	            if j == 0:
	                writer.writerow([i+1, int(k.pt[0]), int(k.pt[1]), str(k.size)[:5], cont_totals[i]])
	            else:
	                writer.writerow(['', int(k.pt[0]), int(k.pt[1]), str(k.size)[:5]])
	    writer.writerow([''])
	    writer.writerow(['key #', 'histogram pair', 'chi-square', 'correlation', 'structural similarity', 'negative flag'])
	    for i,d in hists_compare_chisqr.iteritems():
	        for ii, (key, val) in enumerate(hists_compare_chisqr[i].iteritems()):
	            if ii == 0:
	                writer.writerow([i+1, key, val, hists_compare_corr[i][key], ssims[i][key], flags[i]['no_var_sig']])
	            else:
	                writer.writerow(['', key, val, hists_compare_corr[i][key], ssims[i][key]])
	end = time.time()
	print "total analysis time: " + str(end - start) + " seconds"

	return True

if __name__ == '__main__':
	main()
