from gfpeps import gaussian_fpeps

# hydra config
import hydra
from omegaconf import DictConfig

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf/gfpeps", config_name="default")
def main_app(cfg: DictConfig) -> None:
    return gaussian_fpeps(cfg)

if __name__ == '__main__':
    main_app()