# Efficient Self-Supervised Heterogeneous Graph Representation Learning with Reconstruction 

This repository contains the reference code for the paper "Efficient Self-Supervised Heterogeneous Graph Representation Learning with Reconstruction".

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)

## Installation
* pip install -r requirements.txt 
* unzip the datasets

## Preparation

Pretrained model see >>>[here](saved_model/)<<<.

Configs see >>>[here](args.yaml)<<<.


Important args:
* `--use_pretrain` Test checkpoints
* `--dataset` acm, imdb, dblp, amazon
* `--custom_key` Node: node classification 

## Training
python main.py

