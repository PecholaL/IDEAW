""" Train IDEAW
"""


import yaml
from argparse import ArgumentParser

from .solver import Solver

if __name__ == "__main__":
    parser = ArgumentParser
    parser.add_argument("--load_mode", action="store_true")
    parser.add_argument("--load_opt", action="store_true")
    parser.add_argument("--store_model_path")
    parser.add_argument("--load_model_path")
    parser.add_argument("--summary_steps", default=100, type=int)
    parser.add_argument("--save_steps", default=5000, type=int)
    parser.add_argument("-iterations", default=5, type=int)

    args = parser.parse_args()

    with open("./data/config.yaml") as f:
        config_d = yaml.load(f, Loader=yaml.FullLoader)
    with open("./models/config.yaml") as f:
        config_m = yaml.load(f, Loader=yaml.FullLoader)

    solver = Solver(config_data=config_d, config_model=config_m, args=args)

    if args.iterations > 0:
        solver.train(n_iterations=args.iterations)
