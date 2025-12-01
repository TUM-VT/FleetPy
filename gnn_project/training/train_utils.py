
import os
import traceback

from gnn_project.models.hetero_gat import HeteroGAT
# from gnn_project.models.edge_classifier import RFClassifier
# from gnn_project.models.edge_classifier import XGBClassifier
from gnn_project.training.trainer import Trainer
from gnn_project.dataloaders.gnn_dataloader import GNNDataLoader
# from gnn_project.dataloaders.edge_dataloader import EdgeDataLoader
import torch
import logging

import torch.serialization
from torch_geometric.data.storage import BaseStorage

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    'HeteroGAT': HeteroGAT,
    # TODO support other models
    # 'RandomForest': RFClassifier,
    # 'XGBoost': XGBClassifier,
    # Add other models here as needed
}

DATALOADER_REGISTRY = {
    'GNNDataLoader': GNNDataLoader,
    # TODO support other dataloaders
    # 'EdgeDataLoader': EdgeDataLoader,
    # Add other dataloaders here as needed
}


def train_or_load_model(config, data=None, masks=None):
    """Train a new model or load a saved model based on the configuration.

    Args:
        config: Configuration object with training parameters.
        data: (Optional) Preloaded data. If None, data will be loaded.
        masks: (Optional) Preloaded data masks. If None, masks will be loaded.

    Returns:
        model: Trained or loaded model.
        trainer: Trainer object used for training (None if model was loaded).
    """
    trainer = None
    if should_load_model(config):
        model = load_saved_model(config)
    else:
        if data is None or masks is None:
            data, masks = load_data(config)
        model, trainer = init_model_and_trainer(config, data, masks)
        trainer.train(model)
    return model, trainer


def load_data(config):
    """Load data using the specified dataloader in the configuration.

    Args:
        config: Configuration object with data loading parameters.

    Returns:
        data: Loaded data.
        masks: Data masks for training, validation, and testing.
    """
    try:
        loader_class = DATALOADER_REGISTRY[config.dataloader_type]
    except KeyError:
        raise ValueError(f"Unsupported dataloader type: {config.dataloader_type}")
    loader = loader_class(config)
    data, masks = loader.load_data()
    return data, masks


def init_model_and_trainer(config, data, masks):
    """Initialize the model and trainer based on the configuration.

    Args:
        config: Configuration object with model parameters.
        data: Loaded data.
        masks: Data masks for training, validation, and testing.

    Returns:
        model: Initialized model.
        trainer: Initialized trainer.
    """
    model = build_model(config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    trainer = Trainer(data, masks, optimizer, config)
    return model, trainer


def build_model(config):
    """Build the model based on the configuration.

    Args:
        config: Configuration object with model parameters.

    Returns:
        model: Built model.
    """
    try:
        model_class = MODEL_REGISTRY[config.model_type]
    except KeyError:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    model = model_class(config).to(config.device)
    return model


def should_load_model(config):
    """Determine whether to load a saved model based on the configuration.

    Args:
        config: Configuration object with model loading parameters.

    Returns:
        bool: True if a saved model should be loaded, False otherwise.
    """
    if not config.load_saved_model:
        return False
    if not os.path.exists(config.saved_model_path):
        raise FileNotFoundError(f"Expected checkpoint at {config.saved_model_path}")
    return True


def load_saved_model(config):
    """Load a saved model from the specified path in the configuration.

    Args:
        config: Configuration object with model loading parameters.

    Returns:
        model: Loaded model.
    """
    model = build_model(config)
    # Add BaseStorage class to safe globals for loading
    torch.serialization.add_safe_globals([BaseStorage])
    torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
    try:
        checkpoint = torch.load(
            config.saved_model_path, weights_only=False)
    except Exception as e:
        logger.error('Error loading checkpoint with weights_only=False:', exc_info=e)
        checkpoint = torch.load(config.saved_model_path, weights_only=True)
    logger.info(f"Loading model from {config.saved_model_path}")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_edge_predictions(graph, model, device):
    """Get model predictions for a single graph"""
    model.eval()
    with torch.no_grad():
        # Prepare data dictionaries
        graph = graph.to(device)

        # Prepare input dictionaries
        x_dict = {}
        edge_index_dict = {}
        edge_attr_dict = {}

        # Get node features
        for node_type in graph.node_types:
            x_dict[node_type] = graph[node_type].x

        # Get edge features
        for edge_type in graph.edge_types:
            edge_index = graph[edge_type].edge_index
            edge_attr = graph[edge_type].edge_attr

            edge_index_dict[edge_type] = edge_index.long()  # Ensure int64
            edge_attr_dict[edge_type] = edge_attr

        try:
            logits = model(x_dict, edge_index_dict, edge_attr_dict)
            return torch.sigmoid(logits).cpu().numpy()
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            traceback.print_exc()
            return None
