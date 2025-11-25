"""
run_experiment.py

Simple script to load config, prepare data, train or load the model,
and optionally run evaluation or save outputs.
"""
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from pathlib import Path

from gnn_project.config import Config
from gnn_project.training.train_utils import train_or_load_model, load_data


def main():
    # --------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------
    config = Config(
        ml_data_dir=Path('/Users/hoda_hamdy/Documents/Projects/fleetpy/FleetPy/gnn_project/data'),
        experiment_name='gnn_v1',
        sim_start=0,
        sim_end=2*60*60,
        overwrite_data=False,
        load_saved_model=False,
        log_level='DEBUG',
    )

    print("Running experiment with model:", config.model_type)
    print("Device:", config.device)

    logging.basicConfig(level=config.log_level)

    # --------------------------------------------------
    # 2. Load dataset (optional: allow reuse)
    # --------------------------------------------------
    print("Loading data...")
    data, masks = load_data(config)

    # --------------------------------------------------
    # 3. Train or load model
    # --------------------------------------------------
    print("Training or loading model...")
    model, _ = train_or_load_model(config, data=data, masks=masks)

    # --------------------------------------------------
    # 4. Evaluate model (optional)
    # --------------------------------------------------
    evaluate(model, data, masks, config)

    print("Done!")


# ------------------------------------------------------
# Evaluation function (optional)
# ------------------------------------------------------
def evaluate(model, data, masks, config):
    # TODO: Implement evaluation logic as needed
    print("Evaluating model... (not implemented)")
    pass


if __name__ == "__main__":
    main()
