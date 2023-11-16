""" Train IDEAW
"""


import yaml
from argparse import ArgumentParser

from solver import Solver

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_config", default="config.yaml")
    parser.add_argument("--model_config", default="./model/config.yaml")
    parser.add_argument("--data_config", default="./data/config.yaml")
    parser.add_argument("--pickle_path")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_opt", action="store_true")
    parser.add_argument("--store_model_path")
    parser.add_argument("--load_model_path")
    parser.add_argument("--summary_steps", default=2, type=int)
    parser.add_argument("--save_steps", default=5, type=int)
    parser.add_argument("--iterations", default=5, type=int)

    args = parser.parse_args()

    config_m_path = args.model_config
    config_d_path = args.data_config

    solver = Solver(
        config_data_path=config_d_path, config_model_path=config_m_path, args=args
    )

    if args.iterations > 0:
        solver.train(n_iterations=args.iterations)
