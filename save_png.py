import numpy as np
import os
# Standard Torch Library - Data Augmentation - Network Definition
import scipy.io as sio
from skimage.measure import compare_psnr, compare_ssim
#from scipy.misc import imsave
from common import mriutils
import matplotlib.pyplot as plt

def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())

def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(gt, pred, data_range=gt.max())


if __name__ == '__main__':

    dataset_dir = './results'
    
    eval_datasets = [{'name' : 'coronal_pd',
               'patients': [1,2,3,4,5,6,7,8,9,10],
               'start_slice': 11, 'end_slice': 30,
               'mask': 'masks/COR_PD_iPat4_masks.mat', #  random6_masks_640_368.mat
               'eval_patients' : [11,12,13,14,15,16,17,18,19,20],
               'eval_slices' : [15,20,25],
               'model_name' : 'sense_recon'}]
              
    for dataset in eval_datasets:
         
        print("Evaluating performance for {:s}".format(dataset['name']))
    
        ssim_eval_dataset = []
        psnr_eval_dataset = []
        base_ssim_patient = []
        base_psnr_patient = []
   
        which_view = os.path.join(dataset_dir, dataset['name'])
        
        for k in dataset['eval_patients']:
                
            subject_id = os.path.join(which_view, str(k))
            
            data1 = '{0}/vs-200.mat'.format(subject_id)
            data2 = '{0}/reference.mat'.format(subject_id)
            data3 = '{0}/zero_filling.mat'.format(subject_id)
            
            vs_recon = sio.loadmat(data1)['result_vs']
            gt = np.complex64(sio.loadmat(data2)['reference'])
            zero_filling = sio.loadmat(data3)['result_zf']
            
            base_ssim_patient.append(mriutils.ssim(zero_filling.transpose(-1,0,1), gt.transpose(-1,0,1)))
            base_psnr_patient.append(mriutils.psnr(zero_filling.transpose(-1,0,1), gt.transpose(-1,0,1)))

            for i in range(vs_recon.shape[-1]):
                print('save slice image {}'.format(i)) 
                            
                plt.imsave(os.path.join(subject_id, 'vs_recon_{}.png'.format(i)), abs(vs_recon[...,i]),cmap='gray')
                plt.imsave(os.path.join(subject_id, 'reference_{}.png'.format(i)), abs(gt[...,i]),cmap='gray')
                plt.imsave(os.path.join(subject_id, 'zero-fill_{}.png'.format(i)), abs(zero_filling[...,i]),cmap='gray')
               
         
        
        
        

