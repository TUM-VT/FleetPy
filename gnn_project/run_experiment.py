"""
run_experiment.py

Loads the peak-hours 8-day scenario data, builds/normalizes graphs, and trains
the GNN model. Run from anywhere - paths are resolved relative to this file.
"""
import logging
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gnn_project.config import Config
from gnn_project.training.train_utils import train_or_load_model


def main():
    config = Config(
        ml_data_dir=Path(__file__).resolve().parent / 'data',
        experiment_name='gnn_peak8day',
        scenario_names={'manhattan_case_study': [
            'gnn_peak_2018-11-11', 'gnn_peak_2018-11-12',
            'gnn_peak_2018-11-13', 'gnn_peak_2018-11-14',  # train
            'gnn_peak_2018-11-15', 'gnn_peak_2018-11-17',  # val (weekday + weekend)
            'gnn_peak_2018-11-16', 'gnn_peak_2018-11-18',  # test (weekday + weekend)
        ]},
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        sim_start=25200, sim_end=36000, sim_step=30,
        overwrite_data=False,
        load_saved_model=False,
        log_level='INFO',
        # fresh-processing multiple uncached days concurrently stacks memory per-thread
        # (measured ~9-13GB for a single day) - keep it sequential to bound peak usage
        max_workers=1,
        epochs=50,
    )

    logging.basicConfig(level=config.log_level)
    print("Model:", config.model_type, "| Device:", config.device)

    model, _ = train_or_load_model(config)
    print("Done! Model saved to", config.saved_model_path)


if __name__ == "__main__":
    main()
