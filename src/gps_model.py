import torch
from torch_geometric.nn import Set2Set
from typing import Any, Dict, Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    GELU,
    Sequential,
    LSTM,
    Dropout,
)

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool,global_mean_pool,GraphMultisetTransformer,GlobalAttention
from torch_geometric.nn.attention import PerformerAttention
from gps_data_utils import get_atom_feature_dims,get_bond_feature_dims
from ssma import SSMA

class NodeEncoder(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """
    def __init__(self, emb_dim):
        super().__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        feature_dims = get_atom_feature_dims()
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        """"""
        encoded_features = 0
        for i in range(x.shape[1]):
            encoded_features += self.atom_embedding_list[i](x[:, i])

        x = encoded_features
        return x

class EdgeEncoder(torch.nn.Module):
    """
    The bond Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output edge embedding dimension
    """
    def __init__(self, emb_dim):
        super().__init__()


        self.bond_embedding_list = torch.nn.ModuleList()
        feature_dims = get_bond_feature_dims()
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        """"""
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        edge_attr = bond_embedding
        return edge_attr

class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any],walk_step: int):
        super().__init__()
     
        self.atom_emb = Linear(62,channels-pe_dim)


        self.bond_emb = Linear(21, channels)

        self.pe_lin = Linear(walk_step, pe_dim) 
        self.pe_norm = BatchNorm1d(walk_step)
        self.convs = ModuleList()
        self.use_ssma  = False
        self.max_neighbors_ssma = 4 # 2,3,4
        self.compression_ssma = 1.0 #  0.1, 0.25, 0.5, 0.75, 1.0
        self.attention_ssma = True # True,False
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                GELU(),
                Linear(channels, channels),
            )
            if self.use_ssma:
                aggr = SSMA(in_dim=channels,
                            num_neighbors=self.max_neighbors_ssma,
                            mlp_compression=self.compression_ssma,
                            use_attention=self.attention_ssma,learn_affine=False)
            else:
                aggr = "add"

            mp_module = GINEConv(nn, aggr=aggr,edge_dim=channels)
            if self.use_ssma:
                mp_module.register_propagate_forward_pre_hook(aggr.pre_aggregation_hook)
            conv = GPSConv(channels, mp_module, heads=8,dropout=0.1,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

    


    def forward(self,data):
        # batched_data = data["gps_data"]
        batched_data = data["pyg_data"]
        x_pe = self.pe_norm(batched_data.pe)
        edge_index = batched_data.edge_index
        x = batched_data.x
        edge_attr = batched_data.edge_attr
        batch = batched_data.batch

        x = self.atom_emb(x)
        
        x = torch.cat((x, self.pe_lin(x_pe)), 1)
       
        edge_attr = self.bond_emb(edge_attr)
        
    
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
           
        x = global_mean_pool(x, batch)
        

        return x
    

    



class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class GPSEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.channels = kwargs['gps_channels']
        self.pe_dim = kwargs['pe_dim']
        self.walk = kwargs["walk_step"]
        self.num_layers = kwargs['gps_num_layers']
        self.attn_type = kwargs['attn_type']
        self.attn_kwargs = kwargs['attn_kwargs']
        self.embed_dim = self.channels
        self.model = GPS(channels=self.channels,pe_dim=self.pe_dim,num_layers=self.num_layers,attn_type=self.attn_type,attn_kwargs=self.attn_kwargs,walk_step=self.walk)
    
    def get_embed_dim(self):

        return self.embed_dim

    def get_split_params(self):

        nopt_params, pt_params = [], []
        for k, v in self.named_parameters():
            nopt_params.append(v)
        return nopt_params, pt_params

    def forward(self, data):
        return self.model(data)
    
