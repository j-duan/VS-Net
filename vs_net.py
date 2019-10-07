import matplotlib.pyplot as plt 

import os, visdom 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import transforms as T
from architecture import network
from skimage.measure import compare_psnr, compare_ssim
from data_loader import MRIDataset, load_traindata_path
from torch.utils.data import DataLoader


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())

def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(gt, pred, data_range=gt.max())
  
def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), 
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

def create_image_window(vis, im_shape, title):
    return vis.image(np.ones(im_shape), opts=dict(title=title))


def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.5 every 5000."""
    if epoch % 50 == 0 and epoch > 55:
        for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        print('LR is set to {}'.format(param_group['lr']))
        
        
def create_log(name, model_name):
    
    log_dir = './log/{}'.format(model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    csv_name = os.path.join(log_dir, '{}.csv'.format(name))
    log = '{}_log'.format('name')
    log = open(csv_name, 'w')
    if name == 'train':
        log.write('epoch, iteration, batch, loss, base_psnr, train_psnr, base_ssim, train_ssim, base_nmse, train_nmse\n')
    else:
        log.write('epoch, iteration, batch, loss, base_psnr, train_psnr, base_ssim, train_ssim, base_nmse, train_nmse\n')
    
    return log
    

def test(epoch):
       
    model.eval()  # test mode
    
    data_len = len(data_list['val'])   
    
    for iteration, samples in enumerate(test_loader): 
        print('  iteration {} out of {} in validation'.format(iteration, epoch))
                    
        img_und, img_gt, rawdata_und, masks, sensitivity = samples
     
        img_gt = torch.tensor(img_gt).to(device)
        img_und = torch.tensor(img_und).to(device)
        rawdata_und = torch.tensor(rawdata_und).to(device)
        masks = torch.tensor(masks).to(device)
        sensitivity = torch.tensor(sensitivity).to(device)

        rec = model(img_und, rawdata_und, masks, sensitivity)
        loss = mse(rec, img_gt)

        sense_recon = T.complex_abs(rec).data.to('cpu').numpy()
        sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()
        sense_und = T.complex_abs(img_und).data.to('cpu').numpy()
        
        if iteration % 5 == 0:
            A = sense_und[0]/(sense_und.max())
            B = sense_recon[0]/(sense_recon.max())
            C = sense_gt[0]/(sense_gt.max())            
            vis.image(np.clip(abs(np.c_[A, B, C, C - B]), 0, 1),
                      win=test_image_window, opts=dict(title='test'))
                    
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([loss.item()]), 
                 update='append', win=test_loss_window)
        
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([ssim(sense_gt[0], sense_recon[0])]),
                 update='append', win=test_ssim_window)
        
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([psnr(sense_gt[0], sense_recon[0])]),
                 update='append', win=test_psnr_window)
        
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([nmse(sense_gt[0], sense_recon[0])]),
                 update='append', win=test_nmse_window)
            
        for idx in range(img_gt.shape[0]):
            base_psnr = psnr(abs(sense_gt[idx]), abs(sense_und[idx]))
            base_ssim = ssim(abs(sense_gt[idx]), abs(sense_und[idx]))
            base_nmse = nmse(abs(sense_gt[idx]), abs(sense_und[idx]))
            test_psnr = psnr(abs(sense_gt[idx]), abs(sense_recon[idx]))
            test_ssim = ssim(abs(sense_gt[idx]), abs(sense_recon[idx]))
            test_nmse = nmse(abs(sense_gt[idx]), abs(sense_recon[idx]))
                        
            if idx == 0: 
                val_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'. \
                                format(epoch, iteration, idx, loss.item(), base_psnr, \
                                test_psnr, base_ssim, test_ssim, base_nmse, test_nmse))
                val_log.flush()
            else: 
                val_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'. \
                                format(epoch, '', idx, '', base_psnr, \
                                test_psnr, base_ssim, test_ssim, base_nmse, test_nmse))
                val_log.flush()
                

def train(epoch):
       
    model.train()  # training mode
    
    data_len = len(data_list['train'])
    
    for iteration, samples in enumerate(train_loader): 
        print('  iteration {} out of {} in training'.format(iteration, epoch))
                    
        img_und, img_gt, rawdata_und, masks, sensitivity = samples
     
        img_gt = torch.tensor(img_gt).to(device)
        img_und = torch.tensor(img_und).to(device)
        rawdata_und = torch.tensor(rawdata_und).to(device)
        masks = torch.tensor(masks).to(device)
        sensitivity = torch.tensor(sensitivity).to(device)

        rec = model(img_und, rawdata_und, masks, sensitivity)
        
        loss = mse(rec, img_gt)
             
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                            
        sense_recon = T.complex_abs(rec).data.to('cpu').numpy()
        sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()
        sense_und = T.complex_abs(img_und).data.to('cpu').numpy()
        
        if iteration % 5 == 0:
            A = sense_und[0]/(sense_und.max())
            B = sense_recon[0]/(sense_recon.max())
            C = sense_gt[0]/(sense_gt.max())            
            vis.image(np.clip(abs(np.c_[A, B, C, C - B]), 0, 1),
                      win=train_image_window, opts=dict(title='train'))
                    
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([loss.item()]), 
                 update='append', win=train_loss_window)
        
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([ssim(sense_gt[0], sense_recon[0])]),
                 update='append', win=train_ssim_window)
        
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([psnr(sense_gt[0], sense_recon[0])]),
                 update='append', win=train_psnr_window)
        
        vis.line(X=np.array([iteration+epoch*data_len]), 
                 Y=np.array([nmse(sense_gt[0], sense_recon[0])]),
                 update='append', win=train_nmse_window)
            
        
        for idx in range(img_gt.shape[0]):
            base_psnr = psnr(abs(sense_gt[idx]), abs(sense_und[idx]))
            base_ssim = ssim(abs(sense_gt[idx]), abs(sense_und[idx]))
            base_nmse = nmse(abs(sense_gt[idx]), abs(sense_und[idx]))
            train_psnr = psnr(abs(sense_gt[idx]), abs(sense_recon[idx]))
            train_ssim = ssim(abs(sense_gt[idx]), abs(sense_recon[idx]))
            train_nmse = nmse(abs(sense_gt[idx]), abs(sense_recon[idx]))
                        
            if idx == 0: 
                train_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'. \
                                format(epoch, iteration, idx, loss.item(), base_psnr, \
                                train_psnr, base_ssim, train_ssim, base_nmse, train_nmse))
                train_log.flush()
            else: 
                train_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'. \
                                format(epoch, '', idx, '', base_psnr, \
                                train_psnr, base_ssim, train_ssim, base_nmse, train_nmse))
                train_log.flush()
                


if __name__ == '__main__':
            
    device = 'cuda:0'
    batch_size = 1
    lr = 0.001
    epoch_number = 201
    batch_size = 1
    dccoeff = 0.1 #data consistency layer parameter (learnable)
    wacoeff = 0.1 #weighted average layer parameter (learnable)
    cascade = 5 #stage number 
    num_workers = 10
    acceleration = 4 #acceleration number = 4 fold
    center_fraction = 0.08 #center fraction 
    constrast = 'coronal_pd'
    dataset_dir = '/home/jinming/Desktop/fastRMI/knee_nyu'
    
    model_name = 'sense_recon'

    
    patch_size = 256  
    vis = visdom.Visdom()    
    train_image_window = create_image_window(vis, (1, patch_size, patch_size*5), 'train_image')                                        
    train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')                          
    train_psnr_window = create_plot_window(vis, '#Iterations', 'PSNR', 'Training PSNR')
    train_ssim_window = create_plot_window(vis, '#Iterations', 'SSIM', 'Training SSIM')
    train_nmse_window = create_plot_window(vis, '#Iterations', 'NMSE', 'Training NMSE')
                                     
    test_image_window = create_image_window(vis, (1, patch_size, patch_size*5), 'test_image')                                        
    test_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Testing Loss')                          
    test_psnr_window = create_plot_window(vis, '#Iterations', 'PSNR', 'Testing PSNR')
    test_ssim_window = create_plot_window(vis, '#Iterations', 'SSIM', 'Testing SSIM')
    test_nmse_window = create_plot_window(vis, '#Iterations', 'NMSE', 'Testing NMSE')                                       
                                        
    model = network(dccoeff, wacoeff, cascade).to(device)   
    mse = nn.MSELoss().to(device)   
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    data_list = load_traindata_path(dataset_dir, constrast)
    
    data_list['train'] =  data_list['train']
    data_list['val']   =  data_list['val']
    
    train_dataset = MRIDataset(data_list['train'], acceleration=acceleration, center_fraction=center_fraction)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    test_dataset = MRIDataset(data_list['val'], acceleration=acceleration, center_fraction=center_fraction)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    # Create a logger
    train_log = create_log('train', model_name)
    val_log = create_log('val', model_name)
    
    
    for epoch in range(epoch_number):

        print('Epoch {}'.format(epoch))
          
        train(epoch)
        
        with torch.no_grad():
            test(epoch)
        
        lr_scheduler(optimizer, epoch)
        
        # save model every 50 epoches
        if epoch % 50 == 0 and epoch > 0:
            print('save the model at epoch {}'.format(epoch))
            model_dir = './model/{}'.format(model_name)
            if not (os.path.exists(model_dir)): os.makedirs(model_dir)
            torch.save(model.state_dict(), "{0}/sense_recon_{1:03d}.pth".format(model_dir, epoch)) 
        
        