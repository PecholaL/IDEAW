""" Training implementation for IDEAW
"""

import os
import random
import torch
import torch.nn as nn
import yaml

from data.dataset import AWdataset, get_data_loader, infinite_iter
from models.ideaw import IDEAW


class Solver(object):
    def __init__(self, config_data, config_model, args):
        self.config_data = config_data
        self.config_model = config_model
        self.args = args

        self.get_inf_train_iter()
        self.build_model()

        if args.load_model:
            self.load_model()

    def get_inf_train_iter(self):
        dataset_dir = self.config_data["out_path"]
        self.dataset = AWdataset(dataset_dir)
        self.train_iter = infinite_iter(
            get_data_loader(dataset=self.dataset, batch_size=10, num_workers=0)
        )
        return

    def cc(self, net):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return net.to(device)

    def build_model(self):
        self.model = self.cc(IDEAW(self.config_model))
        print("[IDEAW]model built")
        print(
            "[IDEAW]total parameter count: {}".format(
                sum(x.numel() for x in self.model.parameters())
            )
        )
        self.optimizer = 



    def train(self, n_iterations):
        for iter in range(n_iterations):
            host_audio = next(self.train_iter)
            
