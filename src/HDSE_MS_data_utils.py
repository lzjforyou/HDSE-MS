import torch
import os
import pickle
from rdkit import Chem
from rdkit.Chem import rdchem
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from communities.algorithms import louvain_method,girvan_newman
from torch_geometric.data import Data
import numpy as np
from pyg_features import atom_featurizer,bond_featurizer
from copy import deepcopy
def trans_to_adj(graph): 

    graph.remove_edges_from(nx.selfloop_edges(graph)) 
    nodes = range(len(graph.nodes)) 
    return np.array(nx.to_numpy_array(graph, nodelist=nodes)) 

num = 0
maxlen=0

def compute_posenc_stats(data, pe_types=None, is_undirected=True, hdse=3):
    global num
    global maxlen
    for t in pe_types:
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes
    else:
        N = data.x.shape[0]
    Maxnode = 500
    flag = 0
    if(data.x.shape[0] >= Maxnode):
        print(data)
        raise Exception("Maxnode exceed")
    G = to_networkx(data, to_undirected=True)
    Maxdis = 10
    Maxspd = 30
    SPD = [[0 for j in range(Maxnode)] for i in range(N)]
    dist = [[0 for j in range(Maxnode)] for i in range(N)]
    G_all = G
    if(hdse == 0):
        pass
        _, communities = nxmetis.partition(G, nparts=5)
        print(communities)
        pass
    elif(hdse == 1):
        print(G.nodes(), G.edges())
        adj_matrix = nx.to_numpy_array(G)
        communities = spectral_clustering(adj_matrix, 5)
        print(communities)
        pass
    elif(hdse == 2):
        graph = pygsp.graphs.Graph(nx.adjacency_matrix(G).todense())
        C, Gc, _, _ = coarsen(graph, K=10, r=0.9, method='algebraic_JC')
        dense_matrix = C.toarray()
        row_indices, col_indices = dense_matrix.nonzero()
        max_cluster = np.max(row_indices) + 1
        cluster_list = [[] for _ in range(max_cluster)]
        for i in range(len(row_indices)):
            cluster = col_indices[i]
            point = row_indices[i]
            cluster_list[point].append(cluster)
        communities = cluster_list
        pass
    elif(hdse == 3):
        adj_matrix = nx.to_numpy_array(G)
        communities, _ = girvan_newman(adj_matrix)
    elif(hdse == 4):
        adj_matrix = nx.to_numpy_array(G)
        communities, _ = louvain_method(adj_matrix)
    elif(hdse == 10):
        _, communities = nxmetis.partition(G, nparts=10)
        pass
    M = nx.quotient_graph(G_all, communities, relabel=True)
    dict_graph = {}
    for i in range(len(communities)):
        for j in communities[i]:
            dict_graph[j] = i
    length = dict(nx.all_pairs_shortest_path_length(G_all))
    for i in range(N):
        for j in range(N):
            if(j in length[i]):
                SPD[i][j] = length[i][j]
                if(SPD[i][j] >= Maxspd):
                    SPD[i][j] = Maxspd
                maxlen = max(SPD[i][j], maxlen)
            else:
                SPD[i][j] = Maxspd
    G = M
    length = dict(nx.all_pairs_shortest_path_length(G))
    for i in range(N):
        for j in range(N):
            if(dict_graph[j] in length[dict_graph[i]]):
                dist[i][j] = length[dict_graph[i]][dict_graph[j]]
                if(dist[i][j] >= Maxdis):
                    dist[i][j] = Maxdis
            else:
                dist[i][j] = Maxdis
    laplacian_norm_type = 'none'
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        if 'LapPE' in pe_types:
            max_freqs = 8
            eigvec_norm = "L2"
        elif 'EquivStableLapPE' in pe_types:
            max_freqs = 8
            eigvec_norm = "L2"
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)
        abs_pe = data.EigVecs
    if 'SignNet' in pe_types:
        norm_type = posenc_SignNet.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=norm_type,
                           num_nodes=N)
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=posenc_SignNet.eigen.max_freqs,
            eigvec_norm=posenc_SignNet.eigen.eigvec_norm)
        abs_pe = data.eigvecs_sn
    if 'RWSE' in pe_types:
        kernel_param = list(range(1, 21))
        if len(kernel_param) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing
        abs_pe = rw_landing
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        abs_pe = data.evects_heat
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)
        if 'HKfullPE' in pe_types:
            raise NotImplementedError()
        if 'HKdiagSE' in pe_types:
            kernel_param = posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag
            abs_pe = hk_diag
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic
        abs_pe = elstatic
    SPD = torch.tensor(SPD).long()
    dist = torch.tensor(dist).long()
    complete_edge_index_dist = dist[:N, :N]
    complete_edge_index_dist = complete_edge_index_dist.reshape(-1)
    complete_edge_index_SPD = SPD[:N, :N]
    complete_edge_index_SPD = complete_edge_index_SPD.reshape(-1)
    data.complete_edge_dist = complete_edge_index_dist
    data.complete_edge_SPD = complete_edge_index_SPD
    s = torch.arange(N)
    data.complete_edge_index = torch.vstack((s.repeat_interleave(N), s.repeat(N)))
    return data

def smiles2graph(smiles_or_mol):
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        assert isinstance(smiles_or_mol, rdchem.Mol)
        mol = smiles_or_mol
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

def graph2data(graph, hdse=3):
    data = Data()
    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.float32)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.float32)
    data.y = torch.Tensor([-1])
    pe_types = ['RWSE']
    data = compute_posenc_stats(data, pe_types=pe_types, is_undirected=True, hdse=hdse)
    return data

def HDSE_MS_preprocess(mol, idx):
    graph = smiles2graph(mol)
    data = graph2data(graph)
    data.idx = idx
    return data

def test_pyg_preprocess():
    mol = "C(C(C(=O)O)N)S"
    idx = 0
    data = HiDeeST_MS_preprocess(mol, idx)
    print("Generated Data object:")
    print(data)
    required_attributes = ['x', 'edge_index', 'edge_attr', 'pestat_RWSE', 'complete_edge_dist', 'complete_edge_SPD']
    missing_attributes = [attr for attr in required_attributes if not hasattr(data, attr)]
    if len(missing_attributes) > 0:
        print("Test failed, missing attributes:", missing_attributes)
    else:
        print("Test passed, all expected attributes exist!")
    print("\nDetailed Data object info:")
    print(f"x (node features): {data.x.shape}")
    print(f"edge_index: {data.edge_index.shape}")
    print(f"edge_attr: {data.edge_attr.shape}")
    print(f"pestat_RWSE (RWSE encoding): {data.pestat_RWSE.shape}")
    print(f"complete_edge_dist: {data.complete_edge_dist.shape}")
    print(f"complete_edge_SPD: {data.complete_edge_SPD.shape}")
    print(f"complete_edge_index: {data.complete_edge_index.shape}")


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    N = len(evals)
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)
    return EigVals, EigVecs

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None, num_nodes=None, space_dim=0):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)
    return rw_landing

def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]
        evals = evals.unsqueeze(-1)
        evects = evects.transpose(0, 1)
        eigvec_mul = evects ** 2
        for t in kernel_times:
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul, dim=0, keepdim=False)
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)
    return heat_kernels_diag

def get_heat_kernels(evects, evals, kernel_times=[]):
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]
        evals = evals.unsqueeze(-1).unsqueeze(-1)
        evects = evects.transpose(0, 1)
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))
        for t in kernel_times:
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul, dim=0, keepdim=False)
            )
        heat_kernels = torch.stack(heat_kernels, dim=0)
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)
    return heat_kernels, rw_landing

def get_electrostatic_function_encoding(edge_index, num_nodes):
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)
    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],
        electrostatic.max(dim=0)[0],
        electrostatic.mean(dim=0),
        electrostatic.std(dim=0),
        electrostatic.min(dim=1)[0],
        electrostatic.max(dim=0)[0],
        electrostatic.mean(dim=1),
        electrostatic.std(dim=1),
        (DinvA * electrostatic).sum(dim=0),
        (DinvA * electrostatic).sum(dim=1),
    ], dim=1)
    return green_encoding

def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    EigVals = EigVals.unsqueeze(0)
    if normalization == "L1":
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)
    elif normalization == "L2":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
    elif normalization == "abs-max":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
    elif normalization == "wavelength":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom * 2 / np.pi
    elif normalization == "wavelength-asin":
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = eigval_denom
    elif normalization == "wavelength-soft":
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom
    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")
    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom
    return EigVecs

def spectral_clustering(adj_matrix : np.ndarray, k : int) -> list:
    L = laplacian_matrix(adj_matrix)
    V = eigenvector_matrix(L, k)
    count = 1
    communities = init_communities(len(adj_matrix), k)
    while True:
        C = calc_centroids(V, communities)
        updated_communities = update_assignments(V, C, [set({}) for i in range(k)])
        if updated_communities == communities or count==100:
            break
        count += 1
        communities = updated_communities
    return communities




if __name__ == "__main__":
    test_pyg_preprocess()