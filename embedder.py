import torch
from utils import process
from torch_geometric.utils import degree
class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_dblp4057_mat(args.sc)
            features = process.preprocess_features(features)
            adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        if args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_acm_mat()
            features = process.preprocess_features(features)
            adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        if args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_imdb5k_mat(args.sc)
            features = process.preprocess_features(features)
            adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        if args.dataset == "amazon":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_fraudamazon(args.sc)
            features = process.preprocess_features(features)
            args.ft_size = features[0].shape[1]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]

        if args.dataset in ["dblp", "acm", "imdb","amazon"]:
            adj_list = [adj.to_dense() for adj in adj_list]
            idx_p_list = []
            for adj in adj_list:
                deg_list_0 = []
                idx_p_list_0 = []
                deg_list_0.append(0)
                A_degree = degree(adj.to_sparse()._indices()[0], features.shape[0], dtype=int).tolist()
                out_node = adj.to_sparse()._indices()[1]
                for i in range(features.shape[0]):  # features.shape[0] = nb_nodes
                    deg_list_0.append(deg_list_0[-1] + A_degree[i])
                for j in range(1, args.neighbor_num + 1):
                    random_list = [deg_list_0[i] + j % A_degree[i] for i in range(features.shape[0])]
                    idx_p_0 = out_node[random_list]
                    idx_p_list_0.append(idx_p_0)
                idx_p_list.append(idx_p_list_0)

            adj_list = [process.normalize_graph(adj) for adj in adj_list]
            if args.sparse_adj:
                adj_list = [adj.to_sparse() for adj in adj_list]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
            args.ft_size = features.shape[1]
        self.adj_list = adj_list
        self.idx_p_list = idx_p_list
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)
        self.args = args


