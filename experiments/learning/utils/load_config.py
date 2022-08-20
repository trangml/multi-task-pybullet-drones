import pathlib

from omegaconf import DictConfig
from omegaconf import OmegaConf

def load_config(file_path: str) -> DictConfig:
    file_path = pathlib.Path(file_path)
    cfg_file = file_path.parent / (file_path.stem  + ".yaml")
    cli = OmegaConf.from_cli()
    if "config" in cli:
        file_path = pathlib.Path(cli["config"])
    cfg = OmegaConf.load(cfg_file)
    return OmegaConf.merge(cfg, cli)