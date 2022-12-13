import numpy as np
from scipy import linalg

def calculate_fid(real, generated):
    fid = 0
    num = len(real)
    for index in range(num):
        temp_fid = 0
        for i in range(3):
            # calculate mean and covariance statistics
            mu1, sigma1 = real[index].mean(axis=0), np.cov(real[index][:,:,i], rowvar=False)
            mu2, sigma2 = generated[index].mean(axis=0), np.cov(generated[index][:,:,i], rowvar=False)

            # calculate sum squared difference between means
            ssdiff = np.sum((mu1 - mu2)**2.0)

            # calculate sqrt of product between cov
            covmean = linalg.sqrtm(sigma1.dot(sigma2))

            # check and correct imaginary numbers from sqrt
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            # calculate score
            temp_fid += ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        fid += temp_fid / 3
    return fid/num

# fid = calculate_fid(colors, prediction)
# print('FID: %.3f' % fid)