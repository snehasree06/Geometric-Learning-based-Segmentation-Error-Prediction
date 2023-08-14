import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, GATv2Conv, GCNConv, MessagePassing, SAGEConv
from collections import OrderedDict

from torch.nn import init
import math 
from torch.nn.functional import group_norm
from GNN_models import NodeFormer,NodeFormerConv, GraphSage
from difformer import DIFFormer, DIFFormerConv

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x

class up_conv(nn.Module):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
    def __init__(self, ch_in, ch_out, kernel_size=1, scale=2, align_corners=False):
        super(up_conv, self).__init__()
        self.conv = nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size)
        self.upsample = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners)     
    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        return out

class ResNet_block(nn.Module):
    def __init__(self, ch, kernel_size, stride=1, padding=1):
        super(ResNet_block, self).__init__()
        self.conv1 = conv_block(ch,ch)
        self.conv2 = conv_block(ch,ch)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return out


######################################
############ Processors ##############
######################################

class SplineProcessor(nn.Module):
    def __init__(self, spline_deg, kernel_size, aggr):
        super(SplineProcessor, self).__init__()
        in_channels=32
        out_channels=32
        hidden_channels=32
        
        self.conv1 = SplineConv(in_channels=in_channels, out_channels=hidden_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.nonlin1 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=hidden_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.nonlin2 = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm1d(num_features=hidden_channels)
        
        self.conv3 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.norm5 = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, patch_embs, edge_index, edge_attr):
        ## advanced minibatching here:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html          
        
        # using simple 3 layer GCN with residual connections
        x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index, edge_attr)) + patch_embs)
        x = self.norm3(self.nonlin2(self.conv2(x, edge_index, edge_attr)) + x)
        x = self.norm5(self.conv3(x, edge_index, edge_attr) + x)
        return x

class GATProcessor(nn.Module):
    def __init__(self):
        super(GATProcessor, self).__init__()
        in_channels=32
        out_channels=32
        hidden_channels=32
        #concat: Concatenate layer output or not. If not, layer output is averaged over the heads.
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.nonlin1 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.nonlin2 = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm1d(num_features=hidden_channels)
        
        self.conv3 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.norm5 = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, patch_embs, edge_index, edge_attr):
        ## advanced minibatching here:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html          
        
        # using simple 3 layer GCN with residual connections
        x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index)) + patch_embs)
        x = self.norm3(self.nonlin2(self.conv2(x, edge_index)) + x)
        x = self.norm5(self.conv3(x, edge_index) + x)
        return x
    
class NodeformerProcessor(nn.Module):
    def __init__(self, spline_deg, kernel_size, aggr):
        super(NodeformerProcessor, self).__init__()
        in_channels=32
        out_channels=32
        hidden_channels=32    

        self.conv1_1 = SplineConv(in_channels=in_channels, out_channels=hidden_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.nonlin1_1 = nn.LeakyReLU()
        self.norm1_1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv1_2 = SplineConv(in_channels=in_channels, out_channels=hidden_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.nonlin1_2 = nn.LeakyReLU()
        self.norm1_2 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2_1 = NodeFormerConv(in_channels=hidden_channels, out_channels=hidden_channels, num_heads=8)
        self.nonlin2_1 = nn.LeakyReLU()
        self.norm2_1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2_2 = NodeFormerConv(in_channels=hidden_channels, out_channels=hidden_channels, num_heads=8)
        self.norm2_2 = nn.BatchNorm1d(num_features=hidden_channels)




    def forward(self, patch_embs, edge_index, edge_attr,tau=1.0):

        x = self.conv1_1(patch_embs, edge_index, edge_attr)
        print(x.shape)

        x1= x1.unsqueeze(1)
        x1 = x1.unsqueeze(0)
        x2,_ = self.conv2(x1,edge_index,tau)
        x2 = x2.squeeze(0)
        x2 = x2.squeeze(1)
        # x2 = x2 + x1
        x2 = self.nonlin2(x2)
        x2 = self.norm2(x2)

        x2 = x2.unsqueeze(1)
        x2 = x2.unsqueeze(0)
        x3,_ = self.conv3(x2,edge_index,tau)
        x3 = x3.squeeze(0)
        x3 = x3.squeeze(1)
        # x3 = x3 + x2
        x3 = self.norm3(x3)
        
        return x3




        
##################################################################################
#################pretraining################################################

class patchPredictor(nn.Module):
    def __init__(self):
        super(patchPredictor, self).__init__()
        # define the CNN patch encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True)
        )
        self.pred = nn.Conv3d(in_channels=32, out_channels=2, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, patch):
        # encode the patch
        out = self.encoder(patch)
        # pass it through a prediction layer
        out = self.pred(out)
        # apply GAP
        out = self.pooling(out).squeeze()
        # Logits applied in lossFn
        return out
    
######################################################################################################
############################################ CNN for Vertex Normal predictions ############################

class VertexNormalPredictor_cnn(nn.Module):
    def __init__(self):
        super(VertexNormalPredictor_cnn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(in_features=32, out_features=3)
        self.tanh = nn.Tanh()

    def forward(self, patch):

        out = self.encoder(patch)
        # out = self.pooling(out)
        # out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.tanh(out)
        return out  

########################################################## VAE for CT-subvol reconstruction #################################
# class Encoder(nn.Module):
#     """ Encoder module """
#     def __init__(self,latent_dim):
#         super(Encoder, self).__init__()

#         self.conv1 = conv_block(ch_in=1, ch_out=16, kernel_size=3) 
#         self.MaxPool1 = nn.MaxPool3d(3, stride=2, padding=1) 

#         self.conv2 = conv_block(ch_in=16, ch_out=32, kernel_size=3) 
#         self.MaxPool2 = nn.MaxPool3d(3, stride=3, padding=0) 

#         self.reset_parameters()

#         self.latent_dim = latent_dim
#         self.z_mean = nn.Linear(32, latent_dim)
#         self.z_log_sigma = nn.Linear(32, latent_dim)
#         self.epsilon = torch.normal(size=(1, latent_dim), mean=0, std=1.0, device="cuda")
      
#     def reset_parameters(self):
#         for weight in self.parameters():
#             stdv = 1.0 / math.sqrt(weight.size(0))
#             torch.nn.init.uniform_(weight, -stdv, stdv)

#     def forward(self, x): #x is of shape [batch*n_nodes,1, 5, 5, 5]
        
#         x = self.conv1(x)
#         x = self.MaxPool1(x) 
    
#         x = self.conv2(x)
#         x = self.MaxPool2(x) 
        
#         x_f = torch.flatten(x, start_dim=1)
#         z_mean = self.z_mean(x_f)
#         z_log_sigma = self.z_log_sigma(x_f)
#         z = z_mean + z_log_sigma.exp()*self.epsilon

#         return z, z_mean, z_log_sigma    
    
    
# class Decoder(nn.Module):
#     """ Decoder Module """
#     def __init__(self, latent_dim):
#         super(Decoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.linear_up = nn.Linear(latent_dim, 32*1*1*1)
#         self.relu = nn.ReLU()

#         self.upsize2 = up_conv(ch_in=32, ch_out=16, kernel_size=1,scale=3)
#         self.upsize1 = up_conv(ch_in=16, ch_out=1, kernel_size=1,scale=5/3)
#         self.reset_parameters()
      
#     def reset_parameters(self):
#         for weight in self.parameters():
#             stdv = 1.0 / math.sqrt(weight.size(0))
#             torch.nn.init.uniform_(weight, -stdv, stdv)

#     def forward(self, x):
        
#         x = self.linear_up(x)
#         x = self.relu(x)
#         x = x.view(-1, 32,1,1,1)
#         x = self.upsize2(x) 
#         x = self.upsize1(x) 
#         return x


# class VAE(nn.Module):
#     def __init__(self, latent_dim=32):
#         super(VAE, self).__init__()

#         self.latent_dim = latent_dim
#         self.encoder = Encoder(latent_dim)
#         self.decoder = Decoder(latent_dim)

#         self.reset_parameters()
      
#     def reset_parameters(self):
#         for weight in self.parameters():
#             stdv = 1.0 / math.sqrt(weight.size(0))
#             torch.nn.init.uniform_(weight, -stdv, stdv)

#     def forward(self, x):
        
#         z,z_mean,z_log_sigma = self.encoder(x)
#         z = z.view(-1, 32,1,1,1)

#         # y = self.decoder(z)
#         return z    #, z_mean, z_log_sigma  




######################################### AutoEncoder for CT_subvol reconstruction from masked CT-subvol ############################################

# class MaskEncoder(nn.Module):
#     """ Encoder module """
#     def __init__(self,latent_dim):
#         super(MaskEncoder, self).__init__()

#         self.conv1 = conv_block(ch_in=1, ch_out=16, kernel_size=3) 
#         self.MaxPool1 = nn.MaxPool3d(3, stride=2, padding=1) 

#         self.conv2 = conv_block(ch_in=16, ch_out=32, kernel_size=3) 
#         self.MaxPool2 = nn.MaxPool3d(3, stride=3, padding=0) 

#         self.latent_dim = latent_dim
      
#     def forward(self, x): #x is of shape [batch*n_nodes,1, 5, 5, 5]
        
#         x = self.conv1(x)
#         x = self.MaxPool1(x) 
    
#         x = self.conv2(x)
#         x = self.MaxPool2(x) 
        
#         x_f = torch.flatten(x, start_dim=1)

#         return x_f   
    
    
# class MaskDecoder(nn.Module):
#     """ Decoder Module """
#     def __init__(self, latent_dim):
#         super(MaskDecoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.linear_up = nn.Linear(latent_dim, 32*1*1*1)
#         self.relu = nn.ReLU()

#         self.upsize2 = up_conv(ch_in=32, ch_out=16, kernel_size=1,scale=3)
#         self.upsize1 = up_conv(ch_in=16, ch_out=1, kernel_size=1,scale=5/3)
      
#     def forward(self, x):
        
#         x = self.linear_up(x)
#         x = self.relu(x)
#         x = x.view(-1, 32,1,1,1)
#         x = self.upsize2(x) 
#         x = self.upsize1(x) 
#         return x

# class MaskAE(nn.Module):
#     def __init__(self, latent_dim=32):
#         super(MaskAE, self).__init__()

#         self.latent_dim = latent_dim
#         self.encoder = MaskEncoder(latent_dim)
#         self.decoder = MaskDecoder(latent_dim)

#     def forward(self, x):
        
#         z = self.encoder(x)
#         z = z.view(-1, 32,1,1,1)
#         # y = self.decoder(z)
#         return z

class ReconstructionModel(nn.Module):
    def __init__(self):
        super(ReconstructionModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.decoder(x)
        return x
    
class CGM_general(nn.Module):
    def __init__(self, n_classes, device="cuda", processor="spline", spline_deg=2, kernel_size=5, aggr="mean", mlp_features=128, use_pretrain_encoder=False, pretrained_weights=None,encoder="CNN"):
        super(CGM_general, self).__init__()

        if encoder =="CNN":
            self.encoder = patchPredictor().encoder     
        elif encoder=="ReconCT":
            # self.encoder = VAE() #just upto encoder block
            self.encoder = ReconstructionModel()
        elif encoder=="MaskReconCT":
            # self.encoder = MaskAE()
            self.encoder = ReconstructionModel()
        elif encoder=="VN_CNN":
            self.encoder = VertexNormalPredictor_cnn()     
                                                             
            
        if use_pretrain_encoder==True:
            self.pretrained_weights = pretrained_weights
            self.encoder.load_state_dict(pretrained_weights['model'])

        self.pooling = nn.AdaptiveAvgPool3d(1)

        #GNN processor
        if processor=="spline":
            self.processor = SplineProcessor(spline_deg, kernel_size, aggr)
        elif processor=="GAT":
            self.processor = GATProcessor()
        elif processor=="nodeformer":
            self.processor = NodeFormer(in_channels=32,hidden_channels=32,out_channels=32)
            # self.processor = NodeformerProcessor(in_channels=32,hidden_channels=32,out_channels=32)

        elif processor=="difformer":
            self.processor = DIFFormer(in_channels=32,hidden_channels=32,out_channels=32)
        
        #MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=mlp_features),
            nn.Dropout(p=0.25, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=mlp_features),
            nn.Linear(in_features=mlp_features, out_features=mlp_features),
            nn.Dropout(p=0.25, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=mlp_features),
            nn.Linear(in_features=mlp_features, out_features=n_classes)
        )
        self.device = device

    def forward(self, graph):

        enc_out = self.encoder((torch.unsqueeze(graph.patches_tensor, dim=1)))
        graph.x = torch.squeeze(self.pooling(enc_out))     
        node_embeddings = self.processor(graph.x, graph.edge_index, graph.edge_attr)
        out = self.decoder(node_embeddings)
        return out




    
