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
import sklearn
from sklearn.metrics import precision_score, recall_score, classification_report


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_loaders(args):

#     _, _, test_inds = k_fold_split_train_val_test(20, args.fold_num, seed=220469) #23 ct volumes
    _, val_inds,_ = split_train_val_test(22, seed=220469) #22 ct volumes

    val_data = qaTool_dataset(mesh_dir=args.mesh_dir, signed_distances_dir=args.signed_distances_dir, ct_patches_dir=args.ct_patches_dir, mesh_inds=val_inds, perturbations_to_sample_per_epoch=args.dtspe)
    data_loader = DataLoader(dataset=val_data, batch_size=int(args.batch_size),
                            num_workers=8, 
                            shuffle=False)

    return data_loader


def load_model(args):

    checkpoint = torch.load(args.checkpoint)
    args = checkpoint['args']

    n_classes = 1

    if args.use_pretrain_encoder == True and args.pretrained_model_dir!=None:
        pretrained_weights = torch.load(args.pretrained_model_dir)
        model = CGM_general(n_classes=n_classes, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat,use_pretrain_encoder=args.use_pretrain_encoder,pretrained_weights=pretrained_weights,encoder=args.encoder).to(args.device)
    else:
        model = CGM_general(n_classes=n_classes, processor=args.processor, spline_deg=args.spline_deg, kernel_size=args.kernel_size, aggr=args.aggr, mlp_features=args.decoder_feat,use_pretrain_encoder=False,pretrained_weights=None,encoder=args.encoder).to(args.device)
    print(model)
    model.load_state_dict(checkpoint['model'])
    
    return model


def run_model(args,epoch,model, data_loader, writer):

    model.eval()
    checkpoint_dir = args.checkpoint
    len_orig = len(checkpoint_dir)
    exp_name = checkpoint_dir[len_orig:]
    cm = []
    accuracies = []
    f1_score = []
    precision_score = []
    recall_score = []
    start_iter = time.perf_counter()
    # class_counts = torch.load(args.all_signed_classes_dir)
    # n_classes = class_counts.size(0)
    # n_classes = 1
    val_confusion_matrix = ConfusionMatrix(n_classes = 5)
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
            
        fig = val_confusion_matrix.gen_matrix_fig()
        
        # writer.add_figure('Confusion_matrix', fig, epoch)
        # writer.add_scalar('validation_Accuracy', np.mean(accuracies),epoch)
        # writer.add_scalar('f1_score', np.mean(f1_score),epoch)
        # writer.add_scalar('precision_score', np.mean(precision_score),epoch)
        # writer.add_scalar('Recall_score', np.mean(recall_score),epoch)
        # writer.add_scalar('ROC_AUC_Score',np.mean(roc_auc_score))
        try:
            os.mkdir(args.preds_output_dir)
        except OSError:
            pass
        with open(join(args.preds_output_dir,"output.txt"), "w") as file:
            file.write(f"Confusion Matrix: {fig}\n")
            file.write(f"f1 score: {np.mean(f1_score)}\n")
            file.write(f"Precision: {np.mean(precision_score)}\n")
            file.write(f"Recall: {np.mean(recall_score)}\n")
            file.write(f"Accuracy: {np.mean(accuracies)}\n")
        fig.savefig(join(args.preds_output_dir,"confusion_matrix.png"))

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'f1_score = {f1:.4g}'
                f'Time = {time.perf_counter() - start_iter:.4f}s',

            )
        

def main(args):

    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary')) 
    data_loader = create_data_loaders(args)
    model = load_model(args)
    start_epoch = 0
    for epoch in range(start_epoch, args.num_epochs):
        run_model(args,epoch, model, data_loader,writer)



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
    parser.add_argument('--checkpoint-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--mesh-dir',type=str,help='Path to mesh directory')
    parser.add_argument('--gs-classes-dir',type=str,help='Path to gs classes directory')
    parser.add_argument('--signed-distances-dir',type=str,help='Path to signed distances directory')
    parser.add_argument('--ct-patches-dir',type=str,help='Path to CT patches directory')
    parser.add_argument('--triangles-dir',type=str,help='Path to triangles directory')
    parser.add_argument('--uniform-points-dir',type=str,help='Path to uniform points directory')
    parser.add_argument('--all-signed-classes-dir',type=str,help='Path to all signed classes directory')
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
    # parser.add_argument("--use_model", default="baseline", type=str)
    parser.add_argument("--encoder",default="CNN",type=str)
    parser.add_argument("--use_pretrain_encoder", default=False, type=bool)
    parser.add_argument("--optimizer", default="Adam", type=str)

#     parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--dtspe", default=100, type=int)
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

    