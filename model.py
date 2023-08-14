import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, GATv2Conv, GCNConv, MessagePassing,ClusterGCNConv,WLConv
from torch_geometric.nn import SAGEConv,GATConv,GINConv,RGATConv,ChebConv,ResGatedGraphConv,TransformerConv,TAGConv,NNConv,SuperGATConv
from torch_geometric.nn.norm import GraphNorm
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
        # self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        # self.conv1 = TransformerConv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.nonlin1 = nn.LeakyReLU()
        # self.norm1 = GraphNorm(hidden_channels)
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        # self.conv2 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        # self.conv2 = TransformerConv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        self.nonlin2 = nn.LeakyReLU()
        # self.norm3 = GraphNorm(hidden_channels) 
        self.norm3 = nn.BatchNorm1d(num_features=hidden_channels)
        
        self.conv3 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        # self.conv3 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        # self.conv3 = TransformerConv(in_channels=in_channels, out_channels=hidden_channels, heads=8, concat=False)
        # self.norm5 = GraphNorm(out_channels) 
        self.norm5 = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, patch_embs, edge_index, edge_attr):
        ## advanced minibatching here:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html          
        
        # using simple 3 layer GCN with residual connections
        x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index)) + patch_embs)
        x = self.norm3(self.nonlin2(self.conv2(x, edge_index)) + x)
        x = self.norm5(self.conv3(x, edge_index) + x)
        return x
    


class ChebconvProcessor(nn.Module):
    def __init__(self):
        super(ChebconvProcessor, self).__init__()
        in_channels=32
        out_channels=32
        hidden_channels=32
        #concat: Concatenate layer output or not. If not, layer output is averaged over the heads.
        # self.conv1 = ChebConv(in_channels=in_channels, out_channels=hidden_channels, K=1)
        # self.conv1 = TAGConv(in_channels=in_channels, out_channels=hidden_channels, K=5)
        self.conv1 = ClusterGCNConv(in_channels=in_channels,out_channels=hidden_channels)
        # self.conv1 = WLConv()
        self.nonlin1 = nn.LeakyReLU()
        # self.norm1 = GraphNorm(hidden_channels)
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        # self.conv2 = ChebConv(in_channels=hidden_channels, out_channels=hidden_channels,K=1)
        # self.conv2 = TAGConv(in_channels=hidden_channels, out_channels=hidden_channels,K=5)
        self.conv2 = ClusterGCNConv(in_channels=hidden_channels,out_channels=hidden_channels)
        # self.conv2 = WLConv()
        self.nonlin2 = nn.LeakyReLU()
        # self.norm3 = GraphNorm(hidden_channels)
        self.norm3 = nn.BatchNorm1d(num_features=hidden_channels)
        
        # self.conv3 = ChebConv(in_channels=in_channels, out_channels=hidden_channels,K=1)
        # self.conv3 = TAGConv(in_channels=in_channels, out_channels=hidden_channels,K=5)
        self.conv3 = ClusterGCNConv(in_channels=hidden_channels,out_channels=out_channels)
        # self.conv3 = WLConv()
        # self.norm5 = GraphNorm(out_channels)
        self.norm5 = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, patch_embs, edge_index, edge_attr):
        ## advanced minibatching here:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html          
        
        # using simple 3 layer GCN with residual connections
        x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index)) + patch_embs)
        x = self.norm3(self.nonlin2(self.conv2(x, edge_index)) + x)
        x = self.norm5(self.conv3(x, edge_index) + x)
        return x

class ResGatedGraphConvProcessor(nn.Module):
    def __init__(self):
        super(ResGatedGraphConvProcessor, self).__init__()
        in_channels=32
        out_channels=32
        hidden_channels=32
        #concat: Concatenate layer output or not. If not, layer output is averaged over the heads.
        self.conv1 = ResGatedGraphConv(in_channels=in_channels, out_channels=hidden_channels)
        self.nonlin1 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.conv2 = ResGatedGraphConv(in_channels=in_channels, out_channels=hidden_channels)
        self.nonlin2 = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm1d(num_features=hidden_channels)
        
        self.conv3 = ResGatedGraphConv(in_channels=in_channels, out_channels=hidden_channels)
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
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(NodeformerProcessor, self).__init__()
        # in_channels=32
        # out_channels=32
        # hidden_channels=32

        self.conv1 = NodeFormerConv(in_channels=in_channels, out_channels=hidden_channels, num_heads=8)
        self.nonlin1 = nn.LeakyReLU()
        # self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)

        self.conv2 = NodeFormerConv(in_channels=hidden_channels, out_channels=hidden_channels, num_heads=8)
        self.nonlin2 = nn.LeakyReLU()
        # self.norm2 = nn.BatchNorm1d(num_features=hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)

        self.conv3 = NodeFormerConv(in_channels=hidden_channels, out_channels=hidden_channels, num_heads=8)
        # self.norm3 = nn.BatchNorm1d(num_features=out_channels)
        self.norm3 = GraphNorm(out_channels)




    def forward(self, patch_embs, edge_index, edge_attr,tau=1.0):

        patch_embs = patch_embs.unsqueeze(1)
        patch_embs = patch_embs.unsqueeze(0)
        x,_ = self.conv1(patch_embs, edge_index,tau) 
        x = x.squeeze(0)
        x = x.squeeze(1)
        # x = x + patch_embs
        x = self.nonlin1(x)
        x1 = self.norm1(x)

        # print(x.shape)

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
    

class InceptionBlock(nn.Module):
    # def __init__(self, spline_deg, kernel_size, aggr, in_channels=32, hidden_channels=32, out_channels=32):
    def __init__(self, kernel_size, in_channels=32, hidden_channels=32, out_channels=32):
        super(InceptionBlock, self).__init__()
        
        # self.conv1 = SplineConv(in_channels=in_channels, out_channels=hidden_channels, dim=3, kernel_size=kernel_size, degree=spline_deg, aggr=aggr)
        self.conv1 = ChebConv(in_channels=in_channels, out_channels=hidden_channels, K=kernel_size)
        self.nonlin1 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)
        
        # self.nodeformer = NodeFormer(in_channels=hidden_channels,hidden_channels=hidden_channels,out_channels=out_channels) #can also try using NodeFormerConv but then will have to add squeeze and unsqueeze layers in the forward pass
        # self.transconv = TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=4, concat=False)
        self.supergatconv = SuperGATConv(in_channels=hidden_channels,out_channels=hidden_channels,heads=2,concat=False)
        # self.tagconv = TAGConv(in_channels=in_channels, out_channels=hidden_channels,K=3)
        # self.clusterconv = ClusterGCNConv(in_channels=in_channels,out_channels=hidden_channels)
        # self.gatconv = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=8, concat=False)
        # self.gatconv = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, heads=4, concat=False)
        # self.norm2 = nn.BatchNorm1d(num_features=hidden_channels)

    def forward(self, patch_embs, edge_index, edge_attr):       
        
        # x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index, edge_attr)) + patch_embs)
        # x = self.norm1(self.nonlin1(self.conv1(patch_embs, edge_index)) + patch_embs)
        x = self.conv1(patch_embs,edge_index)
        x = self.supergatconv(x,edge_index)
        x = self.norm1(self.nonlin1(x)+x)
        # x = self.nodeformer(x, edge_index, edge_attr) +x # can play around with residual connection here
        # x = self.gatconv(x,edge_index) + x
        # x = self.transconv(x,edge_index) + x
        # x = self.supergatconv(x,edge_index) + x

        # x = (self.conv1(patch_embs, edge_index, edge_attr)) + patch_embs
        # x = self.norm1(self.nonlin1(self.transconv(x, edge_index,edge_attr)) + x)
        # x = self.clusterconv(x,edge_index)
        return x
    
    
class InceptionLayer(nn.Module):
    # def __init__(self, spline_deg, aggr, in_channels=32, hidden_channels=32, out_channels=32):
    def __init__(self, in_channels=32, hidden_channels=32, out_channels=32):

        super(InceptionLayer, self).__init__()
        """
        Can play around with no. of InceptionBlock with varying filter sizes, for now I have set 4 blocks 
        varying from 1 to 7.
        """
        
        # self.conv_fsize_1 = InceptionBlock(spline_deg, 3, aggr, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #1
        # self.conv_fsize_3 = InceptionBlock(spline_deg, 3, aggr, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #3
        # self.conv_fsize_5 = InceptionBlock(spline_deg, 5, aggr, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #5
        # self.conv_fsize_7 = InceptionBlock(spline_deg, 5, aggr, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #7
        self.conv_fsize_1 = InceptionBlock(3, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #1
        self.conv_fsize_3 = InceptionBlock(3, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #3
        self.conv_fsize_5 = InceptionBlock(5, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #5
        self.conv_fsize_7 = InceptionBlock(5, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels) #7
        self.nonlin = nn.LeakyReLU()


    def forward(self, patch_embs, edge_index, edge_attr):
        x1 = self.conv_fsize_1(patch_embs, edge_index, edge_attr)
        x2 = self.conv_fsize_3(patch_embs, edge_index, edge_attr)
        x3 = self.conv_fsize_5(patch_embs, edge_index, edge_attr)
        x4 = self.conv_fsize_7(patch_embs, edge_index, edge_attr)
        y =  self.nonlin(x1+x2+x3+x4)
        return y
        # return x1+x2+x3+x4 #can try concatenating these features and passing them through a layer reducing the no. of channels from 32*4 to 32
        
        
class Inception(nn.Module):
    # def __init__(self, spline_deg, aggr, num_layers=1, in_channels=32, hidden_channels=32, out_channels=32, weight_sharing=False):
    def __init__(self, num_layers=1, in_channels=32, hidden_channels=32, out_channels=32, weight_sharing=False):

        super(Inception, self).__init__()
        
        self.weight_sharing = weight_sharing
        self.num_layers = num_layers
        
        if weight_sharing:
            # self.layers = InceptionLayer(spline_deg, aggr, in_channels=32, hidden_channels=32, out_channels=32)
            self.layers = InceptionLayer(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)

        else:
            # self.layers = nn.ModuleList(num_layers*[InceptionLayer(spline_deg, aggr, in_channels=32, hidden_channels=32, out_channels=32)])
            self.layers = nn.ModuleList(num_layers*[InceptionLayer(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)])

    def forward(self, patch_embs, edge_index, edge_attr):
        x = patch_embs.clone()

        for idx in range(self.num_layers):
            if self.weight_sharing:
                x = self.layers(x, edge_index, edge_attr)        #+ x #can play around with residual connection

            else:
                x = self.layers[idx](x, edge_index, edge_attr)     #+ x #can play around with residual connection

        return x
        
        
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
        else:
            self.encoder = None
                                                             
        # self.encoder = ReconstructionModel()
   
        if use_pretrain_encoder==True:
            self.pretrained_weights = pretrained_weights
            self.encoder.load_state_dict(pretrained_weights['model'])

        self.pooling = nn.AdaptiveAvgPool3d(1)

        #GNN processor
        if processor=="spline":
            self.processor = SplineProcessor(spline_deg, kernel_size, aggr)
        elif processor=="GAT":
            self.processor = GATProcessor()
            # self.processor = ChebconvProcessor()
        elif processor=="nodeformer":
            self.processor = NodeFormer(in_channels=32,hidden_channels=32,out_channels=32)
            # self.processor = NodeformerProcessor(in_channels=32,hidden_channels=32,out_channels=32)
            
        elif processor=="inception": #inception of spline+nodeformer
            # self.processor = Inception(spline_deg, aggr, num_layers=1, weight_sharing=False)
            self.processor = Inception(num_layers=1, weight_sharing=False)

        elif processor=="resgatedgraph":
            # self.processor = DIFFormer(in_channels=32,hidden_channels=32,out_channels=32)
            # self.processor = GraphSage()
            self.processor = ResGatedGraphConvProcessor()
        else:
            self.processor = None
        
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
        # self.lin = nn.Linear(in_features=1,out_features=32)

    def forward(self, graph): #No ablation

        enc_out = self.encoder((torch.unsqueeze(graph.patches_tensor, dim=1)))
        graph.x = torch.squeeze(self.pooling(enc_out))     
        node_embeddings = self.processor(graph.x, graph.edge_index, graph.edge_attr)
        out = self.decoder(node_embeddings)
        return out
    # def forward(self, graph): #GNN Ablation

    #     enc_out = self.encoder((torch.unsqueeze(graph.patches_tensor, dim=1)))
    #     graph.x = torch.squeeze(self.pooling(enc_out))     
    #     out = self.decoder(graph.x)
    #     return out
    # def forward(self, graph): #CNN ablation
    #     enc_out = torch.unsqueeze(graph.patches_tensor, dim=1)
    #     graph.x = torch.squeeze(self.pooling(enc_out))  
    #     graph.x = self.lin(torch.unsqueeze(graph.x,dim=1))
    #     node_embeddings = self.processor(graph.x, graph.edge_index, graph.edge_attr)
    #     out = self.decoder(node_embeddings)
    #     return out




    
