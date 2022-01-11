"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018
"""

import os
from glob import glob
from multiprocessing import cpu_count, Pool

import numpy as np
from PIL import Image
from pprint import pprint
import prnu
from tqdm import tqdm 
import numpy as np
import cv2

def main():
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    
    DATABASE_PATH = "C:/Users/rlfalsgh95/source/repos/PRNU/PRNU_Business"

    type = "Flatfield"

    ff_dirlist = np.array(sorted(glob(os.path.join(DATABASE_PATH, type, "*", "*.JPG"))))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 2)[0] for i in ff_dirlist])   # ex. Nikon_D70s_1

    nat_dirlist = np.array(sorted(glob(os.path.join(DATABASE_PATH, "*", "natural", "*", "*.JPG"))) + sorted(glob(os.path.join(DATABASE_PATH, "*", "natural", "*.JPG")))) 
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 2)[0] for i in nat_dirlist]) # ex. Nikon_D70s_1

    print("# of flatfield", len(ff_dirlist))
    print("# of natural", len(nat_dirlist))

    print('Computing fingerprints')
    fingerprint_device = sorted(np.unique(ff_device))

    print(ff_dirlist, ff_device)
    K_PATH = os.path.join(DATABASE_PATH, type, "PRNU.npy")
    i = 0
    crop_path = os.path.join(DATABASE_PATH, type, "cropped")

    W_PATH = os.path.join(DATABASE_PATH, "w.npy")
    GT_PATH = os.path.join(DATABASE_PATH, "gt.npy")
    CC_PATH = os.path.join(DATABASE_PATH, "cc.npy")
    STATS_PATH = os.path.join(DATABASE_PATH, "stats.npy")

    if not os.path.exists(K_PATH) : 
        k = []
        for device in tqdm(fingerprint_device): # PRNU 계산
            imgs = []
            for img_path in ff_dirlist[ff_device == device]:
                im = Image.open(img_path)
                im_arr = np.asarray(im)
                if im_arr.dtype != np.uint8:
                    print('Error while reading image: {}'.format(img_path))
                    continue
                if im_arr.ndim != 3:
                    print('Image is not RGB: {}'.format(img_path))
                    continue
                im_cut = prnu.cut_ctr(im_arr, (512, 512, 3)) # 이미지를 center crop
                cv2.imwrite(os.path.join(crop_path, "{}.jpg".format(i)), im_cut)
                i+=1
                imgs += [im_cut] # center crop한 이미지를 list에 추가
            k += [prnu.extract_multiple_aligned(imgs, processes=cpu_count())]

        k = np.stack(k, 0)
        np.save(K_PATH, k)
    else : 
        k = np.load(K_PATH)

    print('Computing residuals')

    return;
    if not os.path.exists(W_PATH) : # Noise 추출
        imgs = []
        for img_path in tqdm(nat_dirlist):
            imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

        pool = Pool(cpu_count())
        w = pool.map(prnu.extract_single, imgs)
        pool.close()

        w = np.stack(w, 0)
        np.save(W_PATH, w)
    else : 
        w = np.load(W_PATH)
    

    # Computing Ground Truth
    if not os.path.exists(GT_PATH) : 
        gt = prnu.gt(fingerprint_device, nat_device)
        np.save(GT_PATH, gt)
    else : 
        gt = np.load(GT_PATH)

    print('Computing cross correlation')
    if not os.path.exists(CC_PATH) : 
        cc_aligned_rot = prnu.aligned_cc(k, w)['cc']
        np.save(CC_PATH, cc_aligned_rot)
    else : 
        cc_aligned_rot = np.load(CC_PATH)

    print('Computing statistics cross correlation')

    if not os.path.exists(STATS_PATH) : 
        stats_cc = prnu.stats(cc_aligned_rot, gt)
        np.save(STATS_PATH, stats_cc)
    else : 
        stats_cc = np.load(STATS_PATH)

    """
    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(tqdm(k)):
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']

    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)
    print('AUC on PCE {:.2f}'.format(stats_pce['auc']))
    """

    print('AUC on CC {:.2f}'.format(stats_cc['auc']))


if __name__ == '__main__':
    main()
