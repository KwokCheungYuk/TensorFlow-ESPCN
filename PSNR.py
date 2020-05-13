import math

import numpy as np


# 计算PSNR
def cal_psnr(img1, img2):
    img1_data = np.asarray(img1).astype('float64')
    img2_data = np.asarray(img2).astype('float64')
    different = img1_data - img2_data
    different = different.flatten('C')
    MSE = np.mean(different ** 2.)
    if MSE == 0:
        return 100
    return 10 * math.log10((255.0 * 255.0)/ MSE)
