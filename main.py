from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1_000)
    return parser.parse_args()