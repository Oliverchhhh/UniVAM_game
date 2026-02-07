import argparse

from dotenv import load_dotenv
from omegaconf import OmegaConf


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/debug.yaml", required=False, type=str)
    args = parser.parse_args()

    load_dotenv()

    OmegaConf.register_new_resolver("oc.load", lambda path: OmegaConf.load(path))
    config = OmegaConf.load(args.config_path)

    return config
