""" Training implementation for IDEAW
"""

import torch
import torch.nn as nn
import yaml

from data.dataset import AWdataset, get_data_loader, infinite_iter
from models.ideaw import IDEAW


class Solver(object):
    def __init__(self, config_data_path, config_model_path, args):
        self.config_data_path = config_data_path
        self.config_model_path = config_model_path
        self.args = args  # training config inside

        with open(self.config_data_path) as f:
            self.config_data = yaml.load(f, Loader=yaml.FullLoader)
        with open(self.config_model_path) as f:
            self.config_model = yaml.load(f, Loader=yaml.FullLoader)
        with open(self.args.train_config) as f:
            self.config_t = yaml.load(f, Loader=yaml.FullLoader)

        self.get_inf_train_iter()  # prepare data
        self.build_model()  # prepare model
        self.build_optims()  # prepare optimizers
        self.loss_criterion()  # prepare criterions

        if self.args.load_model:
            self.load_model()

    # prepare data, called in Solver.init
    def get_inf_train_iter(self):
        dataset_dir = self.args.pickle_path
        self.dataset = AWdataset(dataset_dir)
        self.batch_size = self.config_t["train"]["batch_size"]
        self.num_workers = self.config_t["train"]["num_workers"]
        self.train_iter = infinite_iter(
            get_data_loader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        )
        print("[IDEAW]infinite dataloader built")
        return

    # load model/data to cuda
    def cc(self, net):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return net.to(device)

    # called in Solver.init
    def build_model(self):
        self.model = self.cc(IDEAW(self.config_model_path))
        print("[IDEAW]model built")
        print(
            "[IDEAW]total parameter count: {}".format(
                sum(x.numel() for x in self.model.parameters())
            )
        )

    # called in Solver.init
    def build_optims(self):
        param_hinet1 = list(
            filter(lambda p: p.requires_grad, self.model.hinet_1.parameters())
        )
        param_hinet2 = list(
            filter(lambda p: p.requires_grad, self.model.hinet_2.parameters())
        )
        param_discr = list(
            filter(lambda p: p.requires_grad, self.model.discriminator.parameters())
        )
        param_att = list(
            filter(lambda p: p.requires_grad, self.model.attack_layer.parameters())
        )
        param_balance = list(
            lambda p: p.requires_grad, self.model.balance_block.parameters()
        )

        lr1 = eval(self.config_t["train"]["lr1"])
        lr2 = eval(self.config_t["train"]["lr2"])
        beta1 = self.config_t["train"]["beta1"]
        beta2 = self.config_t["train"]["beta2"]
        eps = eval(self.config_t["train"]["eps"])
        weight_decay = eval(self.config_t["train"]["weight_decay"])

        self.optim_stage_I = torch.optim.Adam(
            param_hinet1 + param_hinet2 + param_discr,
            lr=lr1,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        self.optim_II = torch.optim.Adam(
            param_att+,
            lr=lr2,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        # self.optim_att = torch.optim.Adam(
        #     param_att,
        #     lr=lr3,
        #     betas=(beta1, beta2),
        #     eps=eps,
        #     weight_decay=weight_decay,
        # )
        self.weight_scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optim_inn,
            self.config_t["train"]["weight_step"],
            gamma=self.config_t["train"]["gamma"],
        )
        self.weight_scheduler2 = torch.optim.lr_scheduler.StepLR(
            self.optim_discr,
            self.config_t["train"]["weight_step"],
            gamma=self.config_t["train"]["gamma"],
        )
        # self.weight_scheduler2 = torch.optim.lr_scheduler.StepLR(
        #     self.optim_discr,
        #     self.config_t["train"]["weight_step"],
        #     gamma=self.config_t["train"]["gamma"],
        # )
        print("[IDEAW]optimizers built")

    # autosave, called in training
    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.args.store_model_path}ideaw.ckpt")
        torch.save(
            self.optim_inn.state_dict(), f"{self.args.store_model_path}optim1.opt"
        )
        torch.save(
            self.optim_discr.state_dict(), f"{self.args.store_model_path}optim2.opt"
        )
        # torch.save(
        #     self.optim_att.state_dict(), f"{self.args.store_model_path}optim3.opt"
        # )

    # load trained model
    def load_model(self):
        print(f"[IDEAW]load model from {self.args.load_model_path}")
        self.model.load_state_dict(torch.load(f"{self.args.load_model_path}ideaw.ckpt"))
        self.optim_inn.load_state_dict(
            torch.load(f"{self.args.load_model_path}optim1.opt")
        )
        self.optim_discr.load_state_dict(
            torch.load(f"{self.args.load_model_path}optim2.opt")
        )
        # self.optim_att.load_state_dict(
        #     torch.load(f"{self.args.load_model_path}optim3.opt")
        # )
        return

    # loss criterion
    def loss_criterion(self):
        self.criterion_percept = nn.MSELoss()
        self.criterion_integ = nn.MSELoss()
        self.criterion_discr = nn.BCELoss()

    # training
    def train(self, n_iterations):
        print("[IDEAW]starting training...")
        percept_loss_history = []
        integ_loss_history = []
        discr_loss_history = []
        for iter in range(n_iterations):
            # get data for current iteration
            host_audio = next(self.train_iter).to(torch.float32)
            msg_len = self.config_model["IDEAW"]["num_bit"]
            lcode_len = self.config_model["IDEAW"]["num_lc_bit"]
            watermark_msg = torch.randint(
                0, 2, (self.batch_size, msg_len), dtype=torch.float32
            )
            locate_code = torch.randint(
                0, 2, (self.batch_size, lcode_len), dtype=torch.float32
            )
            orig_label = torch.ones((self.batch_size, 1))
            wmd_label = torch.zeros((self.batch_size, 1))

            ## load to cuda
            host_audio = self.cc(host_audio)
            watermark_msg = self.cc(watermark_msg)
            locate_code = self.cc(locate_code)
            orig_label = self.cc(orig_label)
            wmd_label = self.cc(wmd_label)

            # forward
            (
                audio_wmd1,
                _,
                audio_wmd2,
                _,
                msg_extr1,
                msg_extr2,
                lcode_extr,
                orig_output,
                wmd_output,
            ) = self.model(host_audio, watermark_msg, locate_code)

            # loss
            ## percept. loss
            percept_loss_1 = self.criterion_percept(host_audio, audio_wmd1)
            percept_loss_2 = self.criterion_percept(host_audio, audio_wmd2)
            percept_loss = percept_loss_1 + percept_loss_2
            percept_loss_history.append(percept_loss.item())
            ## integrity loss
            integ_loss_1 = self.criterion_integ(watermark_msg, msg_extr1)
            integ_loss_2 = self.criterion_integ(watermark_msg, msg_extr2)
            integ_loss_3 = self.criterion_integ(locate_code, lcode_extr)
            integ_loss = integ_loss_1 + integ_loss_2 + integ_loss_3
            integ_loss_history.append(integ_loss.item())
            ## discriminate loss
            discr_loss_orig = self.criterion_discr(orig_output, orig_label)
            discr_loss_wmd = self.criterion_discr(wmd_output, wmd_label)
            discr_loss = discr_loss_orig + discr_loss_wmd
            discr_loss_history.append(discr_loss.item())
            ## total loss
            total_loss = percept_loss + integ_loss + discr_loss

            # backward
            total_loss.backward()

            if eval(self.config_t["train"]["optim1_step"]):
                self.optim_inn.step()
            if eval(self.config_t["train"]["optim2_step"]):
                self.optim_discr.step()
            # if eval(self.config_t["train"]["optim3_step"]):
            #     self.optim_att.step()
            self.optim_inn.zero_grad()
            self.optim_discr.zero_grad()
            # self.optim_com.zero_grad()

            # log
            print(
                f"[IDEAW]:[{iter+1}/{n_iterations}]",
                f"loss_percept={percept_loss.item():.6f}",
                f"loss_integ={integ_loss.item():6f}",
                f"loss_discr={discr_loss.item():6f}",
                end="\r",
            )

            # summary
            if (iter + 1) % self.args.summary_steps == 0 or iter + 1 == n_iterations:
                print()

            # autosave
            if (iter + 1) % self.args.save_steps == 0 or iter + 1 == n_iterations:
                self.save_model()

        return
