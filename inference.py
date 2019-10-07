import numpy as np

# Standard Torch Library - Data Augmentation - Network Definition
import torch, os, glob
import scipy.io as sio
from data_loader import load_file, data2complex, cobmine_all_coils
from architecture import network
from data import transforms as T
from common.subsample import MaskFunc
from common.utils import tensor_to_complex_np
from common import mriutils



def data_for_training(rawdata, sensitivity, mask_func, norm=True):
    ''' normalize each slice using complex absolute max value'''
    
    rawdata = T.to_tensor(np.complex64(rawdata.transpose(2,0,1)))
    
    sensitivity = T.to_tensor(sensitivity.transpose(2,0,1))
        
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
     

if __name__ == '__main__':
    
    device  = 'cuda:0'
    dccoeff = 0.1 #data consistency layer parameter (learnable)
    wacoeff = 0.1 #weighted average layer parameter (learnable)
    cascade = 5 #stage number 
    acc = 4 #acceleration number = 4 fold
    center_fract = 0.08 #center fraction 

    dataset_dir = '/home/jinming/Desktop/fastRMI/knee_nyu'
    
    eval_datasets = [{'name' : 'coronal_pd',
               'patients': [1,2,3,4,5,6,7,8,9,10],
               'start_slice': 11, 'end_slice': 30,
               'mask': 'masks/COR_PD_iPat4_masks.mat',
               'eval_patients' : [11,12,13,14,15,16,17,18,19,20],
               'eval_slices' : [15,20,25],
               'model_name' : 'sense_recon'
               }]
             
    with torch.no_grad():
            
        for dataset in eval_datasets:
             
            rec_net = network(dccoeff, wacoeff, cascade).to('cuda')
                
            PATH = './model/{}/sense_recon_200.pth'.format(dataset['model_name'])
            rec_net.load_state_dict(torch.load(PATH))
            rec_net.eval()  # test mode
                        
            print("Evaluating performance for {:s}".format(dataset['name']))
        
            ssim_eval_dataset = []
            psnr_eval_dataset = []
            base_ssmi_eval_dataset = []
            base_psnr_eval_dataset = []
                
            which_view = os.path.join(dataset_dir, dataset['name'])
            
            for k in dataset['eval_patients']:
                    
                subject_id = os.path.join(which_view, str(k))
                n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
                    
                output = []
                target = []
                input0 = []
                normalization = []
                
                for i in range(1, n_slice+1):
                                
                    raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
                    sen = '{0}/espirit{1}.mat'.format(subject_id, i)
                    mask = '{0}/{1}'.format(which_view, dataset['mask'])
                    
                    rawdata = sio.loadmat(raw)['rawdata']
                    if dataset['name'] == 'axial_t2':
                        coil_sensitivities = load_file(sen)
                        coil_sensitivities = data2complex(coil_sensitivities['sensitivities']).transpose(2,1,0)
                    else:   
                        coil_sensitivities = np.complex64(sio.loadmat(sen)['sensitivities'])
                    
                    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
                    img_und, img_gt, rawdata_und, masks, sensitivity = data_for_training(rawdata, coil_sensitivities, mask_func)
                         
                    # add batch dimension 
                    batch_img_und = img_und.unsqueeze(0).to(device)
                    batch_rawdata_und = rawdata_und.unsqueeze(0).to(device)
                    batch_masks = masks.unsqueeze(0).to(device)
                    batch_sensitivities = sensitivity.unsqueeze(0).to(device)
                    
                    # deploy the model
                    rec = rec_net(batch_img_und, batch_rawdata_und, batch_masks, batch_sensitivities)        
            
                    # convert to complex
                    batch_recon = tensor_to_complex_np(rec.to('cpu'))
                    batch_img_und = tensor_to_complex_np(batch_img_und.to('cpu'))
                    img_gt = tensor_to_complex_np(img_gt.to('cpu'))
                
                    # squeeze batch dimension
                    batch_recon = np.squeeze(batch_recon, axis=0)
                    batch_img_und = np.squeeze(batch_img_und, axis=0)

                    output.append(batch_recon)
                    input0.append(batch_img_und)
                    target.append(img_gt)
                    normalization.append(np.max(np.abs(batch_img_und)))

                # postprocess images
                output = np.asarray(output)
                target = np.asarray(target)
                input0 = np.asarray(input0)
   
                # evaluation
                base_ssim_patient = mriutils.ssim(input0, target)
                base_psnr_patient = mriutils.psnr(input0, target)
                ssim_patient = mriutils.ssim(output, target)
                psnr_patient = mriutils.psnr(output, target)
                
                base_ssmi_eval_dataset.append(base_ssim_patient)
                base_psnr_eval_dataset.append(base_psnr_patient)
                ssim_eval_dataset.append(ssim_patient)
                psnr_eval_dataset.append(psnr_patient)
                
                print("    Patient {:d}: {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(k, base_ssim_patient, base_psnr_patient, ssim_patient, psnr_patient))
    
                output_path = './results/{}/{}'.format(dataset['name'], k)
                if not (os.path.exists(output_path)): os.makedirs(output_path)
                
                # save the results
                mriutils.saveAsMat(output,  '%s/vs-%d.mat' % (output_path, 200), 'result_vs',
                          mat_dict={'normalization': np.asarray(normalization)})
                mriutils.saveAsMat(target, '%s/reference.mat' % (output_path), 'reference',
                          mat_dict={'normalization': np.asarray(normalization)})
                mriutils.saveAsMat(input0, '%s/zero_filling.mat'% (output_path), 'result_zf',
                          mat_dict={'normalization': np.asarray(normalization)})
    
            print("  Dataset {:s}: {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(dataset['name'],
                                  np.mean(base_ssmi_eval_dataset), np.std(base_ssmi_eval_dataset),
                                  np.mean(base_psnr_eval_dataset), np.std(base_psnr_eval_dataset),
                                  np.mean(ssim_eval_dataset), np.std(ssim_eval_dataset),
                                  np.mean(psnr_eval_dataset), np.std(psnr_eval_dataset)))          
        
        
        
        
        
        
        
        
        
        
        
        