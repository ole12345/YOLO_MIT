import sys
from pathlib import Path
import hydra
from lightning import Trainer
from omegaconf import OmegaConf

# Get project root path dynamically
project_root = Path.cwd()  # You can modify this to your specific project structure
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup

# Define main function which is called manually
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="value",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quite", False),
        default_root_dir=save_path,
    )

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


# Function to initialize and load the configuration manually
def run_with_config(config_file: str):
    # Initialize Hydra with the config directory and config name
    with hydra.initialize(config_path="config"):  # Adjust the path as needed
        # Compose the config, passing the file argument manually
        cfg = hydra.compose(config_name=config_file)
        
        # Call main with the composed configuration
        main(cfg)

