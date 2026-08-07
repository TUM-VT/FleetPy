
import os
import traceback

from gnn_project.models.hetero_gat import HeteroGAT
# from gnn_project.models.edge_classifier import RFClassifier
# from gnn_project.models.edge_classifier import XGBClassifier
from gnn_project.training.trainer import Trainer
from gnn_project.dataloaders.gnn_dataloader import GNNDataLoader
# from gnn_project.dataloaders.edge_dataloader import EdgeDataLoader
from gnn_project.defaults import RR_EDGE_NAME, VR_EDGE_NAME, RR_EDGE_DIM, VR_EDGE_DIM, MODEL_STATE_DICT
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


def train_or_load_model(config, data=None, masks=None) -> tuple:
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


def load_data(config) -> tuple:
    """Load data using the configured dataloader.

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


def load_feature_names(config, experiment_name=None) -> dict:
    """Load feature names for a given experiment.
    
    Args:
        config: Configuration object with data loading parameters.
        experiment_name: (Optional) Name of the experiment. If None, uses config.experiment_name.
        
    Returns:
        Dictionary mapping node/edge types to their feature names.
    """
    try:
        loader_class = DATALOADER_REGISTRY[config.dataloader_type]
    except KeyError:
        raise ValueError(f"Unsupported dataloader type: {config.dataloader_type}")
    loader = loader_class(config)
    return loader.load_feature_names(experiment_name)


def infer_edge_dims(data) -> tuple:
    """Infer RR/VR raw edge feature dims from real graphs instead of hardcoding them.

    Args:
        data: List of HeteroData graphs (as returned by a dataloader's load_data()).

    Returns:
        (rr_edge_dim, vr_edge_dim)

    Raises:
        ValueError: if no graph in `data` has edges of both types.
    """
    for graph in data:
        rr_dim = graph[RR_EDGE_NAME].edge_attr.shape[-1] if graph[RR_EDGE_NAME].num_edges > 0 else None
        vr_dim = graph[VR_EDGE_NAME].edge_attr.shape[-1] if graph[VR_EDGE_NAME].num_edges > 0 else None
        if rr_dim is not None and vr_dim is not None:
            return int(rr_dim), int(vr_dim)
    raise ValueError(
        "Could not infer rr_edge_dim/vr_edge_dim: no graph in the loaded data has both RR and VR edges."
    )


def init_model_and_trainer(config, data, masks) -> tuple:
    """Initialize the model and trainer based on the configuration.

    Args:
        config: Configuration object with model parameters.
        data: Loaded data.
        masks: Data masks for training, validation, and testing.

    Returns:
        model: Initialized model.
        trainer: Initialized trainer.
    """
    config.rr_edge_dim, config.vr_edge_dim = infer_edge_dims(data)
    model = build_model(config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    trainer = Trainer(data, masks, optimizer, config)
    return model, trainer


def build_model(config) -> torch.nn.Module:
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


def should_load_model(config) -> bool:
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


def load_saved_model(config) -> torch.nn.Module:
    """Load a saved model from the specified path in the configuration.

    Args:
        config: Configuration object with model loading parameters.

    Returns:
        model: Loaded model.
    """
    # Add BaseStorage class to safe globals for loading
    torch.serialization.add_safe_globals([BaseStorage])
    torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
    try:
        checkpoint = torch.load(
            config.saved_model_path, weights_only=False)
    except Exception as e:
        logger.error('Error loading checkpoint with weights_only=False:', exc_info=e)
        checkpoint = torch.load(config.saved_model_path, weights_only=True)

    # no dataset available at pure-inference time, so read dims from the checkpoint
    if RR_EDGE_DIM not in checkpoint or VR_EDGE_DIM not in checkpoint:
        raise KeyError(
            f"Checkpoint at {config.saved_model_path} predates rr_edge_dim/vr_edge_dim "
            "tracking - needs to be retrained, not just reloaded."
        )
    config.rr_edge_dim = checkpoint[RR_EDGE_DIM]
    config.vr_edge_dim = checkpoint[VR_EDGE_DIM]
    model = build_model(config)

    logger.info(f"Loading model from {config.saved_model_path}")
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    return model


def get_edge_predictions(graph, model, device) -> dict:
    """Get model predictions for a single graph"""
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)

        # Prepare input dictionaries
        x_dict = {ntype: graph[ntype].x for ntype in graph.node_types}
        edge_index_dict = {
            et: graph[et].edge_index.long() for et in graph.edge_types
        }
        edge_attr_dict = {
            et: graph[et].edge_attr for et in graph.edge_types
        }
        
        try:
            logits = model(x_dict, edge_index_dict, edge_attr_dict)
            preds = torch.sigmoid(logits).cpu()
            preds_by_edge_type = {}
            start_idx = 0
            for edge_type in graph.edge_types:
                num_edges = graph[edge_type].edge_index.shape[1]
                end_idx = start_idx + num_edges
                preds_by_edge_type[edge_type] = preds[start_idx:end_idx]
                start_idx = end_idx 
            return preds_by_edge_type
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            traceback.print_exc()
            return None
