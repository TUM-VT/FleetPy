# %% [markdown]
# # GNN Training

# %%
import os

import torch

from gnn_model.HeteroGAT import HeteroGAT
from gnn_model.trainer import Trainer
from data_processing.data_loader import DataLoader
from data_processing.config import DataProcessingConfig

# %% [markdown]
# ## Data Preparation

# %%
results_dir = os.path.join('..', 'studies', 'manhattan_case_study', 'results')
# scenarios = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f)) and f != 'archived']
scenario_names = ['gnn_ex1', 'gnn_ex2', 'gnn_ex3', 'gnn_ex4']
scenarios = [os.path.join(results_dir, sc) for sc in scenario_names]
config = DataProcessingConfig(
    sim_duration=86400
)
overwrite = True

# %%

data, masks = DataLoader(scenarios, config, overwrite=overwrite).load_data()

# %% [markdown]
# ## Model

# %%
num_classes = 2
hidden_channels = 64
epochs = 200
batch_size = 32

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
model = HeteroGAT(hidden_channels, num_classes).to(device=device)

# %%
criterion = torch.nn.BCELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

# %%
trainer = Trainer(data, device, masks, config, batch_size=batch_size)
trainer.train(model, criterion, optimizer)

# %%
# TODO hyperparameter tuning
# TODO visualize results


