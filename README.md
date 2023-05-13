# Self-supervised Representation Learning for Heterogeneous Graph

This repository contains the reference code for the paper "Self-supervised Representation Learning for Heterogeneous Graph".

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)
0. [Testing](#test)

## Installation
pip install -r requirements.txt 

## Preparation

Pretrained model see >>>[here](saved_model/)<<<.

Configs see >>>[here](args.yaml)<<<.


Important args:
* `--use_pretrain` Test checkpoints
* `--dataset` acm, imdb, dblp, amazon
* `--custom_key` Node: node classification 

## Training
python main.py

