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

        if args.load_model:
            self.load_model()

    # prepare data, called in Solver.init
    def get_inf_train_iter(self):
        dataset_dir = self.args.pickle_path
        self.dataset = AWdataset(dataset_dir)
        self.train_iter = infinite_iter(
            get_data_loader(dataset=self.dataset, batch_size=10, num_workers=0)
        )
        print("[IDEAW]infinite dataloader built")
        return

    # load model to cuda, called in Solver.init.buildmode
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
        lr1 = self.config_t["train"]["lr1"]
        lr2 = self.config_t["train"]["lr2"]
        beta1 = self.config_model["train"]["beta1"]
        beta2 = self.config_model["train"]["beta2"]
        eps = self.config_model["train"]["eps"]
        weight_decay = self.config["train"]["weight_decay"]
        self.optim_inn1 = torch.optim.Adam(
            param_hinet1,
            lr=lr1,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        self.optim_inn2 = torch.optim.Adam(
            param_hinet2,
            lr=lr2,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        self.weight_scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optim_inn1,
            self.config_t["train"]["weight_step"],
            gamma=self.config_t["train"]["gamma"],
        )
        self.weight_scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optim_inn1,
            self.config_t["train"]["weight_step"],
            gamma=self.config_t["train"]["gamma"],
        )
        print("[IDEAW]optimizers built")

    # autosave, called in training
    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.args.store_model_path}ideaw.ckpt")
        torch.save(
            self.optim_inn1.state_dict(), f"{self.args.store_model_path}optim1.opt"
        )
        torch.save(
            self.optim_inn2.state_dict(), f"{self.args.store_model_path}optim2.opt"
        )

    # load trained model
    def load_model(self):
        print(f"[IDEAW]load model from {self.args.load_model_path}")
        self.model.load_state_dict(torch.load(f"{self.args.load_model_path}ideaw.ckpt"))
        self.optim_inn1.load_state_dict(
            torch.load(f"{self.args.load_model_path}optim1.opt")
        )
        self.optim_inn2.load_state_dict(
            torch.load(f"{self.args.load_model_path}optim2.opt")
        )
        return

    # training
    def train(self, n_iterations):
        print("[IDEAW]starting training...")
        for iter in range(n_iterations):
            # get data for current iteration
            host_audio = next(self.train_iter)
            msg_len = self.config_model["IDEAW"]["num_bit"]
            lcode_len = self.config_model["IDEAW"]["num_lc_bit"]
            tmp_tensor = torch.randint(-1, 2, (msg_len,))
            watermark_msg = tmp_tensor[tmp_tensor == 0] = -1
            locate_code = torch.randint(0, 2, (lcode_len,))

            # forward 1
            audio_wmd1, audio_wmd1_stft = self.model.embed_msg(
                host_audio, watermark_msg
            )
            ## get msg from 1st embedding
            msg_extr1 = self.model.extract_msg(audio_wmd1_stft)

            # forward 2
            audio_wmd2, audio_wmd2_stft = self.model.embed_lcode(
                audio_wmd1_stft, locate_code
            )
            ## get lcode from 2nd embedding
            mid_stft, lcode_extr = self.model.extract_lcode(audio_wmd2_stft)
            ## get msg after extracting lcode
            msg_extr2 = self.model.extract_msg(mid_stft)

            # loss
            percept_loss_history = []
            integ_loss_history = []
            ## percept. loss
            criterion_1 = nn.MSELoss()
            percept_loss_1 = criterion_1(host_audio, audio_wmd1)
            percept_loss_2 = criterion_1(host_audio, audio_wmd2)
            percept_loss = percept_loss_1 + percept_loss_2
            percept_loss_history.append(percept_loss.item())
            ## integrity loss
            criterion_2 = nn.MSELoss()
            integ_loss_1 = criterion_2(watermark_msg, msg_extr1)
            integ_loss_2 = criterion_2(watermark_msg, msg_extr2)
            integ_loss_3 = criterion_2(locate_code, lcode_extr)
            integ_loss = integ_loss_1 + integ_loss_2 + integ_loss_3
            integ_loss_history.append(integ_loss.item())
            ## total loss
            total_loss = percept_loss + integ_loss

            # backward
            total_loss.backward()

            if self.config_t["train"]["optim1_step"]:
                self.optim_inn1.step()
            if self.config_t["train"]["optim2_step"]:
                self.optim_inn2.step()

            self.optim_inn1.zero_grad()
            self.optim_inn2.zero_grad()

            # log
            print(
                f"[IDEAW]:[{iter+1}/{n_iterations}]",
                f"loss_percept={percept_loss:.6f}",
                f"loss_integ={integ_loss:6f}",
                end="\r",
            )

            # autosave
            if (iter + 1) % self.args.save_steps == 0 or iter + 1 == n_iterations:
                self.save_model()
                print()

            return
