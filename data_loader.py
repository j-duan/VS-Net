import glob, os, h5py
import numpy as np
from scipy.io import loadmat
from common.subsample import MaskFunc
from data import transforms as T
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def cobmine_all_coils(image, sensitivity):
    """return sensitivity combined images from all coils""" 
    combined = T.complex_multiply(sensitivity[...,0], 
                                  -sensitivity[...,1], 
                                  image[...,0],  
                                  image[...,1])
    
    return combined.sum(dim = 0)

    
def load_file(fname):
    with h5py.File(fname, 'r') as f:    
        arrays = {}
        for k, v in f.items():
            arrays[k] = np.array(v)
    return arrays


def data2complex(x):
    return x.view(dtype=np.complex128)


class MRIDataset(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch(
            subject_id,
            self.acceleration,
            self.center_fraction
        )


def data_for_training(rawdata, sensitivity, mask_func, norm=True):
    ''' normalize each slice using complex absolute max value'''
    
    coils, Ny, Nx, ps = rawdata.shape
   
    # shift data
    shift_kspace = rawdata
    x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    adjust = (-1) ** (x + y)
    shift_kspace = T.ifftshift(shift_kspace, dim=(-3,-2)) * torch.from_numpy(adjust).view(1, Ny, Nx, 1).float()
 
    # apply masks
    
    shape = np.array(shift_kspace.shape)
    shape[:-3] = 1
    mask = mask_func(shape)
    mask = T.ifftshift(mask)  # shift mask

    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), shift_kspace)
    masks = mask.repeat(coils, Ny, 1, ps)

    img_gt, img_und = T.ifft2(shift_kspace), T.ifft2(masked_kspace)
    
    if norm:
        # perform k space raw data normalization
        # during inference there is no ground truth image so use the zero-filled recon to normalize
        norm = T.complex_abs(img_und).max()
        if norm < 1e-6: norm = 1e-6
        # normalized recon
    else: 
        norm = 1
    
    # normalize data to learn more effectively    
    img_gt, img_und = img_gt/norm, img_und/norm

    rawdata_und = masked_kspace/norm  # faster

    sense_gt = cobmine_all_coils(img_gt, sensitivity)
    
    sense_und = cobmine_all_coils(img_und, sensitivity) 
        
    return sense_und, sense_gt, rawdata_und, masks, sensitivity
     

def get_epoch_batch(subject_id, acc, center_fract):
    ''' get training data '''
    
    rawdata_name, coil_name = subject_id

    rawdata = np.complex64(loadmat(rawdata_name)['rawdata']).transpose(2,0,1)
                   
    #    coil_sensitivities = load_file(coil_name)
    #    coil_sensitivities = data2complex(coil_sensitivities['sensitivities']).transpose(2,1,0)
    #    coil_sensitivities = np.complex64(coil_sensitivities)
    
    sensitivity = np.complex64(loadmat(coil_name)['sensitivities'])
    
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
       
    rawdata2 = T.to_tensor(rawdata)
    
    sensitivity2 = T.to_tensor(sensitivity.transpose(2,0,1))
    
    return data_for_training(rawdata2, sensitivity2, mask_func)


def load_traindata_path(dataset_dir, name):
    """ Go through each subset (training, validation) under the data directory
    and list the file names and landmarks of the subjects
    """
    train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    validation = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
    which_view = os.path.join(dataset_dir, name)
    
    data_list =  {}
    data_list['train'] = []
    data_list['val'] = []
    
    for k in train:
            
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
            
        for i in range(11, 30): #(1, n_slice+1):
    
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['train'] += [[raw, sen]]
        
    
    for k in validation:
            
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
            
        for i in range(11, 30): #(1, n_slice+1):
    
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['val'] += [[raw, sen]]   
            
    return data_list
            
            

if __name__ == '__main__':
    
    name = 'coronal_pd'
    dataset_dir = '/home/jinming/Desktop/fastRMI/knee_nyu'
    data_list = load_traindata_path(dataset_dir, name)
    
    train_dataset = MRIDataset(data_list['train'], acceleration=4, center_fraction=0.08)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=1, num_workers=0)
    
    for iteration, sample in enumerate(train_loader):
        sense_und, sense_gt, rawdata_und, masks, sensitivity = sample
             
 
        