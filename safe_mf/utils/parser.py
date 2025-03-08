import argparse

parser = argparse.ArgumentParser(description="Generic parser for Safe-MF")
parser.add_argument("--exp_names", type=str, nargs="+")
parser.add_argument("--path", default=None, type=str)
parser.add_argument("--file", default=None, type=str)
parser.add_argument("--steps", default=None, type=int, nargs="+")
parser.add_argument("--episodes", default=None, type=int, nargs="+")
parser.add_argument("--num_agents", default=None, type=int)
parser.add_argument("--exec_type", default='train', type=str)
parser.add_argument("--image", default=None, type=str)
parser.add_argument("--run_id", default=None, type=str)
