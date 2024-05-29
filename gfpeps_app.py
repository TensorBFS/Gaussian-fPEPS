from gfpeps import gaussian_fpeps

# hydra config
import hydra
from omegaconf import DictConfig

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    logging.info(cfg)
    return gaussian_fpeps(cfg["gfpeps"])

if __name__ == '__main__':
    main_app()