import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import torch_geometric.transforms as T
from massformer.pyg_features import atom_featurizer,bond_featurizer

def smiles2graph(smiles_or_mol):
   

    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        assert isinstance(smiles_or_mol, rdchem.Mol)
        mol = smiles_or_mol

    # mol = Chem.AddHs(mol)



    node_features = np.array([atom_featurizer(atom) for atom in mol.GetAtoms()])

    x = np.array(node_features, dtype=np.float32)

    num_bond_features = 6
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = bond_featurizer(bond)
            edges_list.append((start, end))
            edge_features_list.append(bond_features)
            edges_list.append((end, start))
            edge_features_list.append(bond_features)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.float32)
    else:   
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.float32)

        

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def graph2data(graph):

    data = Data()
    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.float32)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.float32)
    data.y = torch.Tensor([-1])  

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')  
    data = transform(data)

    return data


def pyg_preprocess(mol,idx):
    graph = smiles2graph(mol) 
    data = graph2data(graph) 
    data.idx = idx
    return data

