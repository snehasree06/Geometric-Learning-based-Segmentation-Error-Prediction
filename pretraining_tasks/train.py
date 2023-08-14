import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse

import torch
import torch_geometric
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from model import *

from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from torchvision.utils import make_grid

from dataset import *

from os.path import join

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR

# from kornia.losses import FocalLoss

from utils import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_datasets(args):
    
    train_inds, val_inds, _ = split_train_val_test(21,seed=220469) #21 ct volumes and 22 uCT volumes

    if args.pretrain_flag == "graph_vn":
        train_data = pretrain_dataset_graph(ct_volume_dir=args.ct_volumes_dir,vertex_normals_dir=args.vertex_normals_dir,
                                    mesh_dir=args.mesh_dir, mesh_inds=train_inds,flag=args.pretrain_flag)
        val_data = pretrain_dataset_graph(ct_volume_dir=args.ct_volumes_dir,vertex_normals_dir=args.vertex_normals_dir,
                                    mesh_dir=args.mesh_dir, mesh_inds=val_inds,flag=args.pretrain_flag)

    else:   

        train_data = pretrain_dataset(ct_volume_dir=args.ct_volumes_dir,vertex_normals_dir=args.vertex_normals_dir,
                                        mesh_dir=args.mesh_dir, mesh_inds=train_inds,flag=args.pretrain_flag)
        val_data = pretrain_dataset(ct_volume_dir=args.ct_volumes_dir,vertex_normals_dir=args.vertex_normals_dir,
                                        mesh_dir=args.mesh_dir, mesh_inds=val_inds,flag=args.pretrain_flag)
        
       
    return val_data, train_data


def create_data_loaders(args):

    val_data, train_data = create_datasets(args)  

    if args.pretrain_flag == "graph_vn": 
        train_loader = torch_geometric.loader.DataLoader(dataset=train_data, batch_size=int(args.batch_size), num_workers=4, shuffle=True)
        val_loader = torch_geometric.loader.DataLoader(dataset=val_data, batch_size=int(args.batch_size), num_workers=4, shuffle=False)

    else: 
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=int(args.batch_size),shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=int(args.batch_size), shuffle=False)
        
    return train_loader, val_loader

def train_epoch(args, epoch, model, data_loader, optimizer, writer,LossFn):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    for iter, data in enumerate(tqdm(data_loader)):

        patch = data['patch'].to(args.device)
        label = data['label'].to(args.device)

        if args.pretrain_flag=="mask_ae":
            output = model(patch)
            loss = LossFn(output, label)
        elif args.pretrain_flag=="vae_recon":
            output = model(patch)
            loss = LossFn(output[0], label)
        elif args.pretrain_flag=="vn":
            output = model(patch)
            loss = torch.mean(1-LossFn(output, label))
        elif args.pretrain_flag=="graph_vn":
            graph = data['graph'].to(args.device)
            output = model(graph)
            loss = torch.mean(1-LossFn(output, label))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )

        logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            f'Loss = {loss.item():.4g} '
            f'Time = {time.perf_counter() - start_iter:.4f}s',
        )
        start_iter = time.perf_counter()

    return loss, time.perf_counter() - start_epoch 




def evaluate(args, epoch, model, data_loader, writer,LossFn):

    model.eval()
    losses = []
    accuracies = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            patch = data['patch'].to(args.device)
            label = data['label'].to(args.device)
            
            if args.pretrain_flag=="mask_ae":
                output = model(patch)
                loss = LossFn(output, label)
            elif args.pretrain_flag=="vae_recon":
                output = model(patch)
                loss = LossFn(output[0], label)
            elif args.pretrain_flag=="vn":
                output = model(patch)
                loss = torch.mean(1-LossFn(output, label))
            elif args.pretrain_flag=="graph_vn":
                graph = data['graph'].to(args.device)
                output = model(graph)
                loss = torch.mean(1-LossFn(output, label))

            losses.append(loss.item())

            
        writer.add_scalar('validation_Loss',np.mean(losses),epoch)
        
    return np.mean(losses), time.perf_counter() - start 


# def visualize(args, epoch, model, data_loader, writer):
    
# #     def save_image(image, tag):
# #         image -= image.min()
# #         image /= image.max()
# #         grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
# #         writer.add_image(tag, grid, epoch)

#     model.eval()
#     with torch.no_grad():
#         for iter, data in enumerate(tqdm(data_loader)):
            
#             patch = data['patch'].to(args.device)
#             label = data['label'].to(args.device)
#             soft_pred = model(patch)
            
#             fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(1, 5, figsize=(15, 3), tight_layout=True)
#             axs = (ax0,ax1,ax2,ax3,ax4)
#             for ax_idx, ax in enumerate(axs):
#                 ax.imshow(patch[ax_idx].astype(float), cmap='Greys_r', vmin=0, vmax=1)
#             label = "On contour" if np.argmax(label) == 0 else "Not on contour"
#             pred = "On contour" if np.argmax(soft_pred) == 0 else "Not on contour"
#             ax2.set_title(f"Label: {label}, Pred: {pred}")
#             writer.add_figure(tag='Patch_pred', figure=fig, global_step=self.num_epoch)


def save_model(args, exp_dir, epoch, model, optimizer,best_validation_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_validation_loss': best_validation_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        
        
def build_model(args):

    if args.pretrain_flag=="vn":
        model = VertexNormalPredictor_cnn().to(args.device)
        LossFn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
    elif args.pretrain_flag=="vae_recon":
        # model = VAE().to(args.device)
        model = ReconstructionModel().to(args.device)
      #LossFn = torch.nn.MSELoss()
        LossFn = torch.nn.L1Loss()
    elif args.pretrain_flag=="mask_ae":
        # model = MaskAE().to(args.device)
        model = ReconstructionModel().to(args.device)

      #LossFn = torch.nn.MSELoss()
        LossFn = torch.nn.L1Loss()
    elif args.pretrain_flag=="graph_vn":
        model = VN_CGM().to(args.device)
        LossFn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
    
    return model, LossFn

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model, LossFn = build_model(args)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer 

def build_optim(args, params):
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(params, args.lr, rho=0.9, eps=1e-06, weight_decay=args.weight_decay, foreach=None,maximize=False)
    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=args.lr, lr_decay=0, weight_decay=args.weight_decay, initial_accumulator_value=0, eps=1e-10, foreach=None)

    return optimizer



def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        #checkpoint, model, optimizer, disc, optimizerD = load_model(args, args.checkpoint)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size = 1
        best_validation_loss= checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model, LossFn = build_model(args)   
        optimizer = build_optim(args, model.parameters())
        #print ("Optmizer initialized")
        best_validation_loss = 1e9 #Inital validation loss
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, validation_loader = create_data_loaders(args)
    #print ("Dataloader initialized")
    
#         # Create learning rate adjustment strategy
    if args.lr_sched == "noS":
        scheduler = None
    elif args.lr_sched == "expS":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.lr_sched == "cosS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    elif args.lr_sched == "stepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    
    for epoch in range(start_epoch, args.num_epochs):
#         scheduler.step(epoch)
        train_loss,train_time = train_epoch(args, epoch, model, train_loader,optimizer,writer,LossFn)
        validation_loss,validation_time = evaluate(args, epoch, model, validation_loader, writer,LossFn)
        if args.lr_sched != "noS":
            scheduler.step()
#         visualize(args, epoch, model, display_loader, writer)

        is_new_best = validation_loss < best_validation_loss
        best_validation_loss = min(best_validation_loss,validation_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_validation_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
            f'TrainLoss = {train_loss:.4g} ' 
            f'TrainTime = {train_time:.4f}s '
            f'validation_loss= {validation_loss:.4g} ' 
            f'validationTime = {validation_time:.4f}s',
        )
    writer.close()
    
def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for patchpredictor model')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--mesh-dir',type=str,help='Path to mesh directory')
    parser.add_argument('--ct-volumes-dir',type=str,help='Path to CT volumes directory')
    parser.add_argument('--vertex-normals-dir',type=str,help='Path to vertex normals directory')
    parser.add_argument('--uniform-points-dir',type=str,help='Path to uniform points directory')

    parser.add_argument('--dataset-type',type=str,help='dataset type')
    parser.add_argument("--aggr", default="add", type=str)
    parser.add_argument("--lr_sched", default="expS", type=str)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--pretrain_flag", default="vn", type=str)

    parser.add_argument("--dtspe", default=25, type=int)
    parser.add_argument("--dropout", default=True, type=lambda x: bool(str2bool(x)))

    
    return parser
    
    

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print (args)
    main(args)