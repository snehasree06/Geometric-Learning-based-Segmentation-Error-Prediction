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
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from model import *
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from torchvision.utils import make_grid
from dataset import qaTool_dataset
import os
from os.path import join

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR

from kornia.losses import FocalLoss
from utils import *
import sklearn.metrics as metrics
from math import sqrt
from utils import product_loss
import sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_datasets(args):
    
#     train_inds, val_inds, _ = k_fold_split_train_val_test(22, args.fold_num, seed=220469) #22 uct volumes
    train_inds, val_inds, _ =  split_train_val_test(21, seed=220469) #22 uct volumes and 21 ct volumes

    train_data = qaTool_dataset(mesh_dir=args.mesh_dir, signed_distances_dir=args.signed_distances_dir , ct_patches_dir=args.ct_patches_dir, mesh_inds=train_inds, perturbations_to_sample_per_epoch=args.dtspe)
    val_data = qaTool_dataset(mesh_dir=args.mesh_dir ,signed_distances_dir=args.signed_distances_dir, ct_patches_dir=args.ct_patches_dir, mesh_inds=val_inds, perturbations_to_sample_per_epoch=args.dtspe)

    return val_data, train_data


def create_data_loaders(args):
    
    val_data, train_data = create_datasets(args)   
    train_loader = DataLoader(dataset=train_data, batch_size=int(args.batch_size), num_workers=8, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=int(args.batch_size), num_workers=8, shuffle=False)
    return train_loader, val_loader

def train_epoch(args, epoch, model, data_loader, optimizer, writer,LossFn):
# def train_epoch(args, epoch, model, data_loader, optimizer, writer):

    model.train()
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    for iter, data in enumerate(tqdm(data_loader)):

        graph = data.to(args.device)
        pred_node_classes = model(graph)
        # pred_node_classes = F.softmax(pred_node_classes, dim=1)
        gt_node_classes = graph.y

        loss = LossFn(pred_node_classes.float(),gt_node_classes.float())
        # calculate accuracy
        acc = (torch.argmax(pred_node_classes, dim=1)==torch.argmax(gt_node_classes,dim=1)).sum() / sum(graph.indiv_num_nodes).item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )
        writer.add_scalar('TrainAccuracy',acc.item(),global_step + iter )

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        #break

    return loss, time.perf_counter() - start_epoch, acc


def evaluate(args, epoch, model, data_loader, writer,LossFn):

    model.eval()
    losses = []
    accuracies = []
    cm = []
    acc = []
    f1_score = []
    precision_score = []
    recall_score = []
    roc_auc_score = []
    start_iter = time.perf_counter()
    # class_counts = torch.load(args.all_signed_classes_dir)
    # n_classes = class_counts.size(0)
    n_classes = 5
    val_confusion_matrix = ConfusionMatrix(n_classes = n_classes)
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            
            fname = data.fname[0]
            graph = data.to(args.device)
            pred_node_classes = model(graph)
            gt_node_classes = graph.y
            loss = LossFn(pred_node_classes.float(), gt_node_classes.float())
            # calculate accuracy
            pred_node_classes = F.softmax(pred_node_classes,dim=1)
            acc = (torch.argmax(pred_node_classes, dim=1)==torch.argmax(gt_node_classes,dim=1)).sum() / sum(graph.indiv_num_nodes).item()
            losses.append(loss.item())
            accuracies.append(acc.item())
            gt_arg = np.argmax(gt_node_classes.detach().cpu().numpy(),axis=1)
            pred_arg = np.argmax(pred_node_classes.detach().cpu().numpy(),axis=1)
            # print(f"gt:{gt_node_classes},pred:{pred_node_classes}")
            # cm.append(sklearn.metrics.multilabel_confusion_matrix(np.argmax(gt_node_classes.cpu().numpy()).flatten(),np.argmax(pred_node_classes.cpu().numpy()).flatten()))
            f1 = sklearn.metrics.f1_score(gt_arg,pred_arg,labels=[0,1,2,3,4],average=None)
            f1_score.append(sklearn.metrics.f1_score(gt_arg,pred_arg,labels=[0,1,2,3,4],average=None))
            precision_score.append(sklearn.metrics.precision_score(gt_arg,pred_arg,labels=[0,1,2,3,4],average=None))
            recall_score.append(sklearn.metrics.recall_score(gt_arg,pred_arg,labels=[0,1,2,3,4],average=None))
            # roc_auc_score.append(sklearn.metrics.roc_auc_score(gt_node_classes.cpu().numpy().flatten(),pred_node_classes.cpu().numpy().flatten(),average=None))
            val_confusion_matrix.update(targets=graph.y.detach().cpu().numpy(), soft_preds=pred_node_classes.detach().cpu().numpy())
            
        fig,normed_confusion_matrix,cm_ = val_confusion_matrix.gen_matrix_fig()
        
        writer.add_figure('Confusion_matrix', fig, epoch)
        writer.add_scalar('validation_Loss',np.mean(losses),epoch)
        writer.add_scalar('validation_Accuracy', np.mean(accuracies),epoch)
        writer.add_scalar('f1_score', np.mean(f1_score),epoch)
        writer.add_scalar('precision_score', np.mean(precision_score),epoch)
        writer.add_scalar('Recall_score', np.mean(recall_score),epoch)
        # writer.add_scalar('ROC_AUC_Score',np.mean(roc_auc_score))
        # try:
        #     os.mkdir(args.preds_output_dir)
        # except OSError:
        #     pass
        # with open(join(args.preds_output_dir,"output.txt"), "w") as file:
        #     file.write(f"Confusion Matrix: {sum(cm)}\n")
        #     file.write(f"f1 score: {sum(f1_score)/len(f1_score)}\n")
        #     file.write(f"Precision: {sum(precision_score)/len(precision_score)}\n")
        #     file.write(f"Recall: {sum(recall_score)/len(recall_score)}\n")
        #     file.write(f"Accuracy: {sum(acc)/len(acc)}\n")
        #     file.write(f"ROC_AUC_score: {sum(roc_auc_score)/len(roc_auc_score)}\n")
        
        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'f1_score = {f1:.4g}'
                f'Time = {time.perf_counter() - start_iter:.4f}s',

            )
        
    return np.mean(losses), time.perf_counter() - start_iter, np.mean(accuracies),normed_confusion_matrix,cm_,fig,np.mean(recall_score)


def save_model(args, exp_dir, epoch, model, optimizer,best_validation_loss,val_confusion_matrix,cm_,best_cm,is_new_best,is_new_best_recall_score):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_validation_loss': best_validation_loss,
            'confusion_matrix': val_confusion_matrix,
            'cm_to_get_n':cm_,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        best_cm.savefig(exp_dir/'best_model_cm.png',dpi=300)
    if is_new_best_recall_score:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_rs_model.pt')
        best_cm.savefig(exp_dir/'best_rs_cm.png',dpi=300)
        
        
        
def build_model(args):


    n_classes = 5
    if args.use_pretrain_encoder == True and args.pretrained_model_dir!=None:
        pretrained_weights = torch.load(args.pretrained_model_dir)
        model = CGM_general(n_classes=n_classes, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat,use_pretrain_encoder=args.use_pretrain_encoder,pretrained_weights=pretrained_weights,encoder=args.encoder).to(args.device)
    else:
        model = CGM_general(n_classes=n_classes, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat,use_pretrain_encoder=False,pretrained_weights=None,encoder=args.encoder).to(args.device)

   
    # LossFn = FocalLoss(alpha=class_weights, gamma=2., reduction='mean')
    # LossFn = nn.NLLLoss()
    LossFn = nn.CrossEntropyLoss()
    
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
        # model = build_model(args)
        optimizer = build_optim(args, model.parameters())
        best_validation_loss = 1e9 #Inital validation loss
        best_recall_score = 0.1
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, validation_loader = create_data_loaders(args)

        # Create learning rate adjustment strategy
    if args.lr_sched == "noS":
        scheduler = None
    elif args.lr_sched == "expS":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.lr_sched == "cosS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif args.lr_sched == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time,train_acc = train_epoch(args, epoch, model, train_loader,optimizer,writer,LossFn)
        validation_loss,validation_time, validation_acc,val_confusion_matrix,cm_,best_cm,recall_score = evaluate(args, epoch, model, validation_loader, writer,LossFn)

        # train_loss,train_time,train_acc = train_epoch(args, epoch, model, train_loader,optimizer,writer)
        # validation_loss,validation_time, validation_acc = evaluate(args, epoch, model, validation_loader, writer)

        scheduler.step()


#         visualize(args, epoch, model, display_loader, writer)

        is_new_best = validation_loss < best_validation_loss
        is_new_best_recall_score = recall_score>best_recall_score
        best_recall_score = max(best_recall_score,recall_score)
        best_validation_loss = min(best_validation_loss,validation_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_validation_loss,val_confusion_matrix,cm_,best_cm,is_new_best,is_new_best_recall_score)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
            f'TrainLoss = {train_loss:.4g} ' 
            f'Train_acc={train_acc:.4f}'
            f'TrainTime = {train_time:.4f}s '
            f'validation_loss= {validation_loss:.4g} ' 
            f'validation_acc={validation_acc:.4g} ' 
            f'validationTime = {validation_time:.4f}s',
        )
    writer.close()
    
def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for QATool')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument("--batch-size", default=16, type=int)
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
    # parser.add_argument('--gs-classes-dir',type=str,help='Path to gs classes directory')
    parser.add_argument('--signed-distances-dir',type=str,help='Path to signed distances directory')
    parser.add_argument('--ct-patches-dir',type=str,help='Path to CT patches directory')
    parser.add_argument('--triangles-dir',type=str,help='Path to triangles directory')
    parser.add_argument('--uniform-points-dir',type=str,help='Path to uniform points directory')
    # parser.add_argument('--all-signed-classes-dir',type=str,help='Path to all signed classes directory')
    parser.add_argument('--preds-output-dir',type=str,help='Path to directory where predicted node classes are to be saved')
    parser.add_argument('--pretrained-model-dir',type=str,help='Path to directory where pretrained model weights are to be saved')
    
    parser.add_argument('--dataset-type',type=str,help='dataset type')
    
    parser.add_argument("--processor", default="spline", type=str, help="The type of processor to use")
    parser.add_argument("--decoder_feat", default=128, type=int)
    parser.add_argument("--decoder_bn", default=True, type=lambda x: bool(str2bool(x)))
    parser.add_argument("--spline_deg", default=1, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--aggr", default="add", type=str)
    parser.add_argument("--lr_sched", default="expS", type=str)
    # parser.add_argument("--use_model", default="cnn", type=str)
    parser.add_argument("--encoder", default="cnn", type=str)   
    parser.add_argument("--use_pretrain_encoder", default=False, type=bool)
    parser.add_argument("--optimizer", default="Adam", type=str)

#     parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--dtspe", default=100, type=int) #dtspe - before 100
    # parser.add_argument("--dtspe", default=100, type=int) #dtspe - before 100

    parser.add_argument("--dropout", default=True, type=lambda x: bool(str2bool(x)))

#     parser.add_argument("--encAbl", default=False, type=lambda x: bool(str2bool(x)))
#     parser.add_argument("--GNNAbl", default=False, type=lambda x: bool(str2bool(x)))
#     parser.add_argument("--preAbl", default=True, type=lambda x: bool(str2bool(x)))
    
    return parser
    
    

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
#     print (args)
    main(args)