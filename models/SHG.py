import os
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate
from embedder import embedder
import numpy as np
import random as random
import torch

import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)


class SHG(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.cfg = args.cfg
        self.sigm = nn.Sigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#
        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        print("Started training...")
        model = trainer(self.args)
        model = model.to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        model.train()

        for epoch in tqdm(range(self.args.nb_epochs+1)):
            optimiser.zero_grad()
            loss = model(features, adj_list, self.idx_p_list, epoch)

            loss.backward()
            optimiser.step()
#         torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key))
        if self.args.use_pretrain:
            model.load_state_dict(torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key)))
        print('loss', loss)
        print("Evaluating...")
        model.eval()
        hf = model.embed(features, adj_list)
        macro_f1s, micro_f1s = evaluate(hf, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
        return macro_f1s, micro_f1s

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(2,args.cfg[-1], args.cfg[-1], args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, emb):
        recons = self.linear1(emb)
        recons = self.linear2(F.relu(recons))
        return recons

class trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = args.cfg
        view_num = args.view_num
        self.w_list = nn.ModuleList([nn.Linear(cfg[-1], cfg[-1], bias=True) for _ in range(view_num)])
        self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(view_num)])
        self.W = nn.Parameter(torch.zeros(size=(view_num * cfg[-1], cfg[-1])))
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(self.args.view_num):
            self.encoder.append(make_mlplayers(args.ft_size, args.cfg))
            self.decoder.append(Decoder(args))

    def decode(self, embedding_list):
        recons = []
        for i in range(self.args.view_num):
            tmp = self.decoder[i](embedding_list[i])
            recons.append(tmp)

        return recons

    def combine_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h

    def forward(self, x, adj_list=None, idx_p_list=None, epoch=0):
        x = F.dropout(x, self.args.dropout, training=self.training)
        x_list = []
        for i in range(self.args.view_num):
            x_list.append(x)
        h_p_list = []
        for i in range(self.args.view_num):
            h_a = self.encoder[i](x)
            if self.args.sparse:
                h_p = torch.spmm(adj_list[i], h_a)
            else:
                h_p = torch.mm(adj_list[i], h_a)
            h_p_list.append(h_p)
        recons = self.decode(h_p_list)
        loss_intra = 0
        loss_inv = 0
        recons_err, recons_nei = loss_matching_recons(recons, x_list, idx_p_list, self.args, epoch)
        for i in range(self.args.view_num):
            intra_c = (h_p_list[i]).T @ (h_p_list[i])
            intra_c = F.normalize(intra_c, p=2, dim=1)
            on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
            off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
            loss_intra += (on_diag_intra + self.args.lambdintra[i] * off_diag_intra)
            if i == 1 and self.args.view_num == 2:
                break
            inter_c = (h_p_list[(i + 1) % self.args.view_num]).T @ (h_p_list[i])
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_inv += -torch.diagonal(inter_c).sum()
        loss =  loss_inv + self.args.alpha * loss_intra + self.args.beta * (recons_err + recons_nei)
        return loss

    def embed(self, x, adj_list=None):
        h_p_list = []
        # embedding = []
        for i in range(self.args.view_num):
            h_a = self.encoder[i](x)
            if self.args.sparse:
                h_p = torch.spmm(adj_list[i], h_a)
            else:
                h_p = torch.mm(adj_list[i], h_a)
            h_p_list.append(h_p)
        h_fusion = self.combine_att(h_p_list)

        return h_fusion.detach()

def loss_matching_recons(x_hat, x, idx_p_list, args, epoch):
    l = torch.nn.MSELoss(reduction='sum')

    recons_err = 0
    # Feature reconstruction loss
    for i in range(args.view_num):
        recons_err += l(x_hat[i], x[i])
    recons_err /= x[0].shape[0]

    # Topology reconstruction loss
    interval = int(args.neighbor_num / args.sample_neighbor)
    neighbor_embedding = []
    for i in range(args.view_num):
        neighbor_embedding_0 = []
        for j in range(0, args.sample_neighbor + 1):
            neighbor_embedding_0.append(x[i][idx_p_list[i][(epoch + interval * j) % args.neighbor_num]])
        neighbor_embedding.append(sum(neighbor_embedding_0) / args.sample_neighbor)
    recons_nei = 0
    for i in range(args.view_num):
        recons_nei += l(x_hat[i], neighbor_embedding[i])
    recons_nei /= x[0].shape[0]

    return recons_err, recons_nei


def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
            # result = nn.Sequential(*layers)
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)#, result

class Linearlayer(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Linearlayer, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
