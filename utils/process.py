import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data, ClusterData,NeighborSampler
from torch_geometric.utils import is_undirected, to_undirected
from sklearn.preprocessing import OneHotEncoder
import torch as th
import pickle
from scipy.sparse import coo_matrix

def load_acm_mat(sc=3):
    data = sio.loadmat('data/acm.mat')
    label = data['label']

    adj_edge1 = data["PLP"]
    adj_edge2 = data["PAP"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion

def load_aminer_hetsann(sc=3):
    target_node = [1]
    data = sio.loadmat('data/Aminer_processed.mat')
    node_label_list = ['PvsC', 'AvsC']
    # PAP = data['PAP']
    # PVP = data['PVP']
    # AvsA = data['AvsA']
    # P_features = sp.csr_matrix(data['PvsF'])
    # A_features = sp.csr_matrix(data['AvsF'])

    # adj_fusion1 = PAP + PVP
    # adj_fusion = adj_fusion1.copy()
    # adj_fusion[adj_fusion < 2] = 0
    # adj_fusion[adj_fusion == 2] = 1

    adj1 = data["APA"] + np.eye(data["APA"].shape[0])*sc
    adj2 = data["APVPA"] + np.eye(data["APVPA"].shape[0])*sc
    # adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = adj1

    adj_list = [adj1, adj2]

    truefeatures = sp.csr_matrix(data['AvsF']).astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    # idx_train = data['train_idx'].ravel()
    # idx_val = data['val_idx'].ravel()
    # idx_test = data['test_idx'].ravel()
    number_node = adj1.shape[0]
    random_split = np.random.permutation(number_node)
    idx_train = random_split[:int(number_node * 0.2)]
    idx_val = random_split[int(number_node * 0.2):int(number_node * 0.3)]
    idx_test = random_split[int(number_node * 0.3):]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    label = []
    for t in target_node:
        if t < len(node_label_list):
            label.append(data[node_label_list[t]].toarray())
        else:
            print("type %s node have not label" % (t))
            exit(0)

    return adj_list, truefeatures, label[0], idx_train, idx_val, idx_test, adj_fusion

def load_imdb5k_mat(sc=3):
    data = sio.loadmat('data/imdb5k.mat')
    label = data['label']

    adj_edge1 = data['MAM']
    adj_edge2 = data['MDM']
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1

    adj1 = data['MAM'] + np.eye(data['MAM'].shape[0])*sc
    adj2 = data['MDM'] + np.eye(data['MDM'].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_dblp4057_mat(sc=3):
    data = sio.loadmat('data/DBLP4057.mat')
    label = data['label']

    adj_edge1 = data['net_APTPA']
    adj_edge2 = data['net_APCPA']
    adj_edge3 = data['net_APA']
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data['net_APTPA'] + np.eye(data['net_APTPA'].shape[0])*sc
    adj2 = data['net_APCPA'] + np.eye(data['net_APCPA'].shape[0])*sc
    adj3 = data['net_APA'] + np.eye(data['net_APA'].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['features'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion

def load_dblp(sc=3):
    data = pkl.load(open("data/dblp.pkl", "rb"))
    label = data['label']

    adj_edge1 = data["PAP"]
    adj_edge2 = data["PPrefP"]
    adj_edge3 = data["PATAP"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*sc
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3



    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_imdb(sc=3):
    data = pkl.load(open("data/imdb.pkl", "rb"))
    label = data['label']
###########################################################
    adj_edge1 = data["MDM"]
    adj_edge2 = data["MAM"]
    adj_fusion1 = adj_edge1 + adj_edge2
    # for i in range(0,3550):
    #     for j in range(0, 3550):
    #         if adj_fusion[i][j]!=2:
    #             adj_fusion[i][j]=0
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1
    ############################################################
    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*sc
    adj_fusion = adj_fusion  + np.eye(adj_fusion.shape[0])*3
    # torch.where(torch.eq(torch.Tensor(adj1), 1) == True)
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    # adj1_dense = torch.dense(adj1)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_imdb_2():
    All_data = load_data('IMDB', 0.2)  # Reading Data
    Edges = All_data[0]  # Loading edges, for ACM are PA, AP, PC, CP respectively
    Features = All_data[1]  # Loading features
    valid_node = All_data[3]  # The index of nodes for training
    valid_target = All_data[4]  # The labels of nodes for training

    # IMDB meta-path: MDM, MAM
    # if args.dataset == 'IMDB':
    Graph_1 = graphAugment(generate_meta_path(Edges[0], Edges[1]))
    Graph_2 = graphAugment(generate_meta_path(Edges[2], Edges[3]))
    adj_list = [Graph_1,Graph_2]
    train_node, train_target, test_node, test_target = divide_data(All_data[5], 0.2)
    return adj_list, Features, label, idx_train, idx_val, idx_test, adj_fusion

def divide_data(data, ratio, CUDA_flag=True):
    import random
    length = data.shape[0]
    data = data[random.sample(range(length), length)]
    train_data = data[0:int(ratio*length)]
    test_data = data[int(ratio*length):]

    if CUDA_flag == True:
        train_node = torch.from_numpy(train_data[:, 0]).type(torch.LongTensor).cuda()
        train_target = torch.from_numpy(train_data[:, 1]).type(torch.LongTensor).cuda()
        test_node = torch.from_numpy(test_data[:, 0]).type(torch.LongTensor).cuda()
        test_target = torch.from_numpy(test_data[:, 1]).type(torch.LongTensor).cuda()
    else:
        train_node = torch.from_numpy(train_data[:, 0]).type(torch.LongTensor)
        train_target = torch.from_numpy(train_data[:, 1]).type(torch.LongTensor)
        test_node = torch.from_numpy(test_data[:, 0]).type(torch.LongTensor)
        test_target = torch.from_numpy(test_data[:, 1]).type(torch.LongTensor)

    return train_node, train_target, test_node, test_target

def load_data(dataset, ratio, CUDA_flag=True):      # 数据集名称，训练集占比，是否使用CUDA
    import random
    with open('data/' + dataset + '/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)      #点特征
    with open('data/' + dataset + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)      # 四段元路径 如:PA,AP,PS,SP
    with open('data/' + dataset + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)     # 标签

    labels_ = np.concatenate((labels[0], labels[2]), axis=0)
    node_num = labels_.shape[0]
    labels_ = labels_[random.sample(range(node_num), node_num)]
    test_labels = labels_[int(node_num*ratio):]

    if CUDA_flag == True:
        A = []
        for i, edge in enumerate(edges):
            A.append(torch.from_numpy(edge.todense()).type(torch.FloatTensor).cuda())
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor).cuda()
        valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor).cuda()
        valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor).cuda()
    else:
        A = []
        for i, edge in enumerate(edges):
            A.append(torch.from_numpy(edge.todense()).type(torch.FloatTensor))
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)

    num_classes = torch.max(torch.from_numpy(test_labels[:, 1]).type(torch.LongTensor)).item() + 1
    return A, node_features, num_classes, valid_node, valid_target, labels_

def graphAugment(g):        # 生成D^1/2AD^1/2矩阵
    x, y = g.shape
    #diagl = torch.zeros(x, y)       #draft3
    diagl = torch.zeros(x, y).cuda()       #draft
    for i in range(x):
        if torch.sum(g[i]) != 0.:
            diagl[i][i] = torch.pow(torch.sum(g[i]), -0.5)
    return torch.mm(torch.mm(diagl, g), diagl)

def generate_meta_path(x, y):       # 使得元路径矩阵中大于1的元素等于1
    z = x.mm(y)
    z_ones = torch.ones_like(z)
    z = torch.where(z > 1.0, z_ones, z)
    return z


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot



def load_freebase(sc=3):
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_m = sp.eye(type_num)
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    adj_list = [mam, mdm, mwm]
    adj_fusion = mam
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return  adj_list, feat_m, label, train[0], val[0], test[0], adj_fusion

def load_aminer(sc=3):
    # The order of node types: 0 p 1 a 2 r
    ratio = [20, 40, 60]
    type_num = 6564
    path = "data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    # nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    # nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num)
    # feat_a = sp.eye(type_num[1])
    # feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    # nei_a = [th.LongTensor(i) for i in nei_a]
    # nei_r = [th.LongTensor(i) for i in nei_r]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    # feat_a = th.FloatTensor(preprocess_features(feat_a))
    # feat_r = th.FloatTensor(preprocess_features(feat_r))
    # pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    # prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    # pos = sparse_mx_to_torch_sparse_tensor(pos)
    adj_list = [pap, prp]
    adj_fusion = pap
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return adj_list, feat_p, label, train[0], val[0], test[0], adj_fusion

def load_yelp(sc=3):
    from dgl.data import FraudYelpDataset
    dataset = FraudYelpDataset(random_seed=717, train_size=0.7, val_size=0.1)
    g = dataset[0]
    num_classes = dataset.num_classes
    features = g.ndata['feature']
    labels = g.ndata['label']
    rsr = g.adj(etype='net_rsr')
    rtr = g.adj(etype='net_rtr')
    rur = g.adj(etype='net_rur')
    labels = labels.squeeze()
    features = sp.lil_matrix(features)
    number_node = labels.size()[0]
    random_split = np.random.permutation(number_node)
    idx_train = random_split[:int(number_node * 0.2)]
    idx_val = random_split[int(number_node * 0.2):int(number_node * 0.3)]
    idx_test = random_split[int(number_node * 0.3):]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    rsr_adj = coo_matrix(
        (np.ones(len(rsr._indices()[1])), (rsr._indices()[0].numpy(), rsr._indices()[1].numpy())),
        shape=(rsr.shape[0], rsr.shape[0]))
    I = coo_matrix(
        (np.ones(features.shape[0]), (np.arange(0, features.shape[0], 1), np.arange(0, features.shape[0], 1))),
        shape=(features.shape[0], features.shape[0]))
    rsr_adj = rsr_adj + I
    rsr = sparse_mx_to_torch_sparse_tensor(normalize_adj(rsr_adj))
    rtr_adj = coo_matrix(
        (np.ones(len(rtr._indices()[1])), (rtr._indices()[0].numpy(), rtr._indices()[1].numpy())),
        shape=(rtr.shape[0], rtr.shape[0]))
    rtr_adj = rtr_adj + I
    rtr = sparse_mx_to_torch_sparse_tensor(normalize_adj(rtr_adj))
    rur_adj = coo_matrix(
        (np.ones(len(rur._indices()[1])), (rur._indices()[0].numpy(), rur._indices()[1].numpy())),
        shape=(rur.shape[0], rur.shape[0]))
    rur_adj = rur_adj + I
    rur = sparse_mx_to_torch_sparse_tensor(normalize_adj(rur_adj))
    adj_list = [rsr, rtr, rur]
    adj_fusion = rsr
    labels = encode_onehot(labels)
    return adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion

def load_fraudamazon(sc=3):
    from dgl.data import FraudAmazonDataset
    dataset = FraudAmazonDataset(random_seed=717, train_size=0.7, val_size=0.1)
    g = dataset[0]
    num_classes = dataset.num_classes
    features = g.ndata['feature']
    labels = g.ndata['label']
    rsr = g.adj(etype='net_upu')
    rtr = g.adj(etype='net_usu')
    rur = g.adj(etype='net_uvu')
    labels = labels.squeeze()
    features = sp.lil_matrix(features)
    number_node = labels.size()[0]
    random_split = np.random.permutation(number_node)
    idx_train = random_split[:int(number_node * 0.05)]
    idx_val = random_split[int(number_node * 0.05):int(number_node * 0.1)]
    idx_test = random_split[int(number_node * 0.1):]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    rsr_adj = coo_matrix(
        (np.ones(len(rsr._indices()[1])), (rsr._indices()[0].numpy(), rsr._indices()[1].numpy())),
        shape=(rsr.shape[0], rsr.shape[0]))
    I = coo_matrix(
        (np.ones(features.shape[0]), (np.arange(0, features.shape[0], 1), np.arange(0, features.shape[0], 1))),
        shape=(features.shape[0], features.shape[0]))
    rsr_adj = rsr_adj + sc * I
    rsr = sparse_mx_to_torch_sparse_tensor(normalize_adj(rsr_adj))
    rtr_adj = coo_matrix(
        (np.ones(len(rtr._indices()[1])), (rtr._indices()[0].numpy(), rtr._indices()[1].numpy())),
        shape=(rtr.shape[0], rtr.shape[0]))
    rtr_adj = rtr_adj + sc * I
    rtr = sparse_mx_to_torch_sparse_tensor(normalize_adj(rtr_adj))
    rur_adj = coo_matrix(
        (np.ones(len(rur._indices()[1])), (rur._indices()[0].numpy(), rur._indices()[1].numpy())),
        shape=(rur.shape[0], rur.shape[0]))
    rur_adj = rur_adj + sc * I
    rur = sparse_mx_to_torch_sparse_tensor(normalize_adj(rur_adj))
    adj_list = [rsr, rtr, rur]
    adj_fusion = rsr
    labels = encode_onehot(labels)
    return adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion

def load_amazon(sc=3):
    data = pkl.load(open("data/amazon.pkl", "rb"))
    label = data['label']

    adj_edge1 = data["IVI"]
    adj_edge2 = data["IBI"]
    adj_edge3 = data["IOI"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


