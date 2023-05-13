import argparse
import numpy as np
np.random.seed(0)
from ruamel.yaml import YAML
import os
from models import SHG
import shutil

def get_args(model_name, dataset, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument('--cfg', type=int, default=[128], help='hidden dimension')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
    parser.add_argument('--sparse_adj', type=bool, default=False, help='sparse adjacency matrix')
    parser.add_argument('--iterater', type=int, default=10, help='iterater')
    parser.add_argument('--use_pretrain', type=bool, default=True, help='use_pretrain')
    parser.add_argument('--nb_epochs', type=int, default=1000, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--feature_drop', type=int, default=0.1, help='dropout of features')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=4, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='the seed to use')
    parser.add_argument('--test_epo', type=int, default=100, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
    parser.add_argument('--dropout', type=int, default=0.2, help='dropout')
    parser.add_argument("--neighbor_num", default=300, help="Number of all sampled neighbor",
                        type=int)
    parser.add_argument("--sample_neighbor", default=30, help="Number of sampled neighbor during each iteration",
                        type=int)
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument('--lambdintra', type=float,  help='weight on off-diagonal terms')
    parser.add_argument('--beta', type=float, default=0.001, help='non-negative parameter')
    parser.add_argument('--alpha', default=1, type=float, metavar='L',help='non-negative parameter')
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

def main():
    args = get_args(
        model_name="SHG",
        dataset="imdb", #acm, imdb, dblp, amazon
        custom_key="Node",  # Node: node classification
    )
    if args.dataset == "acm" or args.dataset == "imdb":
        args.view_num = 2
    else:
        args.view_num = 3
    printConfig(args)
    embedder = SHG(args)
    macro_f1s, micro_f1s = embedder.training()

    return macro_f1s, micro_f1s


if __name__ == '__main__':
    macro_f1s, micro_f1s = main()
