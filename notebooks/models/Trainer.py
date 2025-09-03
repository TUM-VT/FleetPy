import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from data_processing.config import DataProcessingConfig as cfg
from typing import Optional, List, Dict
from enum import Enum
import random
import logging


# logging.basicConfig(level=cfg.LOG_LEVEL)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    NONE = "none"  # No special sampling, use standard DataLoader
    DYNAMIC = "dynamic"  # Uses hard example mining
    BALANCED = "balanced"  # Uses class-balanced sampling


# TODO debug
class BalancedBatchSampler:
    """Samples batches with balanced class distribution at edge level within each graph."""

    def __init__(self, data: List, batch_size: int, edge_percentage: float = 0.2):
        """
        Initialize the sampler with percentage-based edge sampling.
        
        Args:
            data: List of graph data
            batch_size: Number of graphs per batch
            edge_percentage: Percentage of edges to sample from each graph (default: 0.2)
        """
        self.data = data
        self.batch_size = batch_size
        self.edge_percentage = edge_percentage
        self.graph_info = self._analyze_graphs()
        self.pos_weighted_indices = self._create_weighted_indices(
            positive=True)
        self.neg_weighted_indices = self._create_weighted_indices(
            positive=False)

    def _analyze_graphs(self) -> Dict[int, Dict]:
        """Analyze each graph's edge label distribution and store edge indices by class."""
        graph_info = {}
        for idx, graph in enumerate(self.data):
            pos_edges = []
            neg_edges = []
            total_edges = 0

            # Store edge indices by their class for each edge type
            for edge_type in graph.y_dict:
                labels = graph.y_dict[edge_type]
                edge_indices = torch.arange(len(labels))

                pos_mask = labels == 1
                neg_mask = labels == 0

                pos_indices = edge_indices[pos_mask].tolist()
                neg_indices = edge_indices[neg_mask].tolist()

                pos_edges.extend([(edge_type, idx) for idx in pos_indices])
                neg_edges.extend([(edge_type, idx) for idx in neg_indices])
                total_edges += len(labels)

            if total_edges > 0:  # Only include graphs with edges
                graph_info[idx] = {
                    'pos_edges': pos_edges,
                    'neg_edges': neg_edges,
                    'total_edges': total_edges,
                    'pos_count': len(pos_edges),
                    'neg_count': len(neg_edges),
                    'pos_ratio': len(pos_edges) / total_edges if total_edges > 0 else 0,
                    'neg_ratio': len(neg_edges) / total_edges if total_edges > 0 else 0
                }

        return graph_info

    def _create_weighted_indices(self, positive: bool) -> List[int]:
        """Create a list of indices weighted by their positive or negative edge counts."""
        weighted_indices = []
        for idx, info in self.graph_info.items():
            # Add the index based on the count of edges of each class
            count = info['pos_count'] if positive else info['neg_count']
            if count > 0:
                weighted_indices.append(idx)
        return weighted_indices

    def _calculate_edges_for_graph(self, graph_idx: int) -> int:
        """Calculate number of edges to sample based on the percentage of total edges."""
        info = self.graph_info[graph_idx]
        total_edges = info['total_edges']
        return max(10, int(total_edges * self.edge_percentage))

    def _sample_balanced_edges_from_graph(self, graph_idx: int) -> Dict[str, List[int]]:
        """Sample a balanced set of edges from a single graph."""
        info = self.graph_info[graph_idx]
        total_edges = self._calculate_edges_for_graph(graph_idx)
        edges_per_class = total_edges // 2  # Split evenly between positive and negative

        sampled_edges = {}
        for edge_type in self.data[graph_idx].y_dict.keys():
            pos_edges_type = [(et, idx) for et, idx in info['pos_edges'] if et == edge_type]
            neg_edges_type = [(et, idx) for et, idx in info['neg_edges'] if et == edge_type]

            # Sample from positive edges
            pos_sample_size = min(len(pos_edges_type), edges_per_class)
            pos_samples = random.sample(pos_edges_type, pos_sample_size) if pos_sample_size > 0 else []

            # Sample from negative edges
            neg_sample_size = min(len(neg_edges_type), edges_per_class)
            neg_samples = random.sample(neg_edges_type, neg_sample_size) if neg_sample_size > 0 else []

            if pos_samples or neg_samples:
                sampled_edges[edge_type] = {
                    'pos': [idx for _, idx in pos_samples],
                    'neg': [idx for _, idx in neg_samples]
                }

        return sampled_edges

    def _create_balanced_graph(self, graph_idx: int, sampled_edges: Dict[str, Dict[str, List[int]]]):
        """Create a new graph with balanced edge samples."""
        original_graph = self.data[graph_idx]
        balanced_graph = original_graph.clone()  # Create a shallow copy

        # Get edge types from original graph
        edge_types = list(original_graph.edge_index_dict.keys())

        # Process each edge type
        for edge_type in edge_types:
            if edge_type in sampled_edges and sampled_edges[edge_type]:
                pos_indices = sampled_edges[edge_type]['pos']
                neg_indices = sampled_edges[edge_type]['neg']

                # Ensure we have at least some edges
                if pos_indices or neg_indices:
                    all_indices = pos_indices + neg_indices

                    # Update edge indices
                    if len(all_indices) > 0:
                        edge_index = original_graph.edge_index_dict[edge_type]
                        if edge_index.size(1) > 0:  # Check if there are any edges
                            balanced_graph.edge_index_dict[edge_type] = edge_index[:, all_indices]

                            # Update edge attributes if they exist
                            if edge_type in original_graph.edge_attr_dict:
                                edge_attr = original_graph.edge_attr_dict[edge_type]
                                if edge_attr is not None and len(edge_attr) > 0:
                                    balanced_graph.edge_attr_dict[edge_type] = edge_attr[all_indices]

                            # Update labels
                            new_labels = torch.zeros(len(all_indices), dtype=torch.float)
                            new_labels[:len(pos_indices)] = 1.0
                            balanced_graph.y_dict[edge_type] = new_labels
            else:
                # If no edges were sampled for this type, create empty tensors
                balanced_graph.edge_index_dict[edge_type] = torch.zeros((2, 0), dtype=torch.long)
                if edge_type in original_graph.edge_attr_dict:
                    attr_size = original_graph.edge_attr_dict[edge_type].size(1)
                    balanced_graph.edge_attr_dict[edge_type] = torch.zeros((0, attr_size))
                balanced_graph.y_dict[edge_type] = torch.zeros(0, dtype=torch.float)

        return balanced_graph

    def sample_batch_indices(self) -> list:
        """Sample a balanced batch and create balanced graphs."""
        if not self.pos_weighted_indices or not self.neg_weighted_indices:
            # Fallback to random sampling if either class is empty
            selected_indices = random.sample(list(self.graph_info.keys()),
                                             min(self.batch_size, len(self.graph_info)))
        else:
            # Sample graphs that have both positive and negative edges
            available_indices = list(set(self.pos_weighted_indices) & set(self.neg_weighted_indices))
            if not available_indices:
                available_indices = list(self.graph_info.keys())

            selected_indices = random.sample(available_indices,
                                             min(self.batch_size, len(available_indices)))

        # Create balanced versions of the selected graphs
        balanced_graphs = []
        for idx in selected_indices:
            sampled_edges = self._sample_balanced_edges_from_graph(idx)
            balanced_graph = self._create_balanced_graph(idx, sampled_edges)
            balanced_graphs.append(balanced_graph)

        # Update the data with balanced graphs
        for i, idx in enumerate(selected_indices):
            self.data[idx] = balanced_graphs[i]

        return selected_indices


# TODO debug
class DynamicBatchSampler:
    def __init__(self, data, batch_size: int, hard_mining_ratio: float = 0.5):
        """
        Initialize dynamic batch sampler with hard example mining.

        Args:
            data: List of graph data objects
            batch_size: Size of each batch
            hard_mining_ratio: Ratio of hard examples to include in each batch
        """
        self.data = data
        self.batch_size = batch_size
        self.hard_mining_ratio = hard_mining_ratio
        self.sample_weights = None
        self.hard_indices = []
        self.easy_indices = []
        self.loss_history = {}  # Keep track of losses for each sample

    def update_mining_weights(self, indices: list, losses: torch.Tensor):
        """Update the loss history and recalculate mining weights."""
        for idx, loss in zip(indices, losses):
            self.loss_history[idx] = loss.item()

        # Sort indices by loss
        sorted_indices = sorted(self.loss_history.items(),
                                key=lambda x: x[1], reverse=True)
        n_hard = int(len(sorted_indices) * self.hard_mining_ratio)

        self.hard_indices = [idx for idx, _ in sorted_indices[:n_hard]]
        self.easy_indices = [idx for idx, _ in sorted_indices[n_hard:]]

    def sample_batch_indices(self) -> list:
        """Sample a batch with a mix of hard and easy examples."""
        n_hard = int(self.batch_size * self.hard_mining_ratio)
        n_easy = self.batch_size - n_hard

        # Handle empty indices case
        if not self.hard_indices:
            if not self.easy_indices:
                # If both empty (e.g., first batch), initialize with all indices
                all_indices = list(range(len(self.data)))
                return random.sample(all_indices, min(self.batch_size, len(all_indices)))
            else:
                # If only hard indices are empty, use all easy indices
                n_easy = self.batch_size

        if not self.easy_indices:
            # If only easy indices are empty, use all hard indices
            n_hard = self.batch_size

        # Sample indices
        hard_samples = np.random.choice(
            self.hard_indices or [0],
            size=min(n_hard, len(self.hard_indices or [])),
            replace=len(self.hard_indices or [0]) < n_hard
        )

        easy_samples = np.random.choice(
            self.easy_indices or [0],
            size=min(n_easy, len(self.easy_indices or [])),
            replace=len(self.easy_indices or [0]) < n_easy
        )

        # If we're missing samples due to empty lists, adjust by sampling from the other list
        missing = self.batch_size - len(hard_samples) - len(easy_samples)
        additional_samples = []

        if missing > 0:
            if len(self.hard_indices or []) > len(hard_samples):
                # Sample more from hard if available
                additional_hard = np.random.choice(
                    [i for i in self.hard_indices if i not in hard_samples],
                    size=min(missing, len(self.hard_indices) -
                             len(hard_samples)),
                    replace=False
                )
                additional_samples.extend(additional_hard)
                missing -= len(additional_hard)

            if missing > 0 and len(self.easy_indices or []) > len(easy_samples):
                # Sample more from easy if needed and available
                additional_easy = np.random.choice(
                    [i for i in self.easy_indices if i not in easy_samples],
                    size=min(missing, len(self.easy_indices) -
                             len(easy_samples)),
                    replace=False
                )
                additional_samples.extend(additional_easy)

        # Combine all samples
        batch_indices = list(hard_samples) + \
                        list(easy_samples) + list(additional_samples)
        random.shuffle(batch_indices)
        return batch_indices


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        """
        Implementation of Focal Loss with alpha balancing.
        Args:
            alpha: Weighting factor for the rare class (default 0.25)
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted (default 2.0)
            pos_weight: Optional tensor of positive weights for balancing
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.eps = 1e-7

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss with alpha balancing.

        Args:
            inputs: Raw logits from the model
            targets: Binary target values (0 or 1)

        Returns:
            Focal loss value
        """
        # Get probabilities with numerical stability
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        # Compute p_t (probability for target class)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Compute alpha_t (alpha weight for target class)
        if self.pos_weight is not None:
            alpha_t = targets * self.pos_weight + (1 - targets)
        else:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Compute modulating factor
        focal_weight = (1 - p_t).pow(self.gamma)

        # Compute CE loss
        ce_loss = -torch.log(p_t)

        # Combine all terms
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()


class Trainer:
    def __init__(self, data, device, masks, config: Optional[cfg] = None, batch_size=32, epochs=200,
                 pos_weight: Optional[float] = None, sampling_strategy: SamplingStrategy = SamplingStrategy.DYNAMIC,
                 hard_mining_ratio: float = 0.5, balance_ratio: float = 1.0):
        """
        Initialize the trainer with configurable sampling strategy

        Args:
            data: Input data
            device: torch device
            masks: Train/val/test masks
            config: Configuration object
            batch_size: Batch size for training
            epochs: Number of training epochs
            pos_weight: Optional weight for positive class
            sampling_strategy: Strategy for batch sampling (DYNAMIC or BALANCED)
            hard_mining_ratio: Ratio of hard examples in dynamic sampling
            balance_ratio: Ratio of negative to positive samples in balanced sampling
        """
        self.device = device
        self.config = config or cfg()
        self.model_dir = os.path.join(
            self.config.base_data_dir, self.config.models_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = 0.5
        self.sampling_strategy = sampling_strategy
        self.hard_mining_ratio = hard_mining_ratio
        self.balance_ratio = balance_ratio

        # Create data loaders
        self.train_loader = self._create_loader(
            data, masks[0], batch_size, shuffle=True)
        self.val_loader = self._create_loader(
            data, masks[1], batch_size, shuffle=False)
        self.test_loader = self._create_loader(
            data, masks[2], batch_size, shuffle=False)

        # Calculate class weights from training data
        if pos_weight is None:
            self.pos_weight = self._calculate_pos_weight(data, masks[0])
        else:
            self.pos_weight = torch.tensor(pos_weight, device=self.device)

        logger.debug(f"\n{'=' * 80}")
        logger.debug(f"Training Configuration")
        logger.debug(f"{'-' * 80}")
        logger.debug(f"{'Batch Size:':<20} {batch_size}")
        logger.debug(f"{'Max Epochs:':<20} {epochs}")
        logger.debug(f"{'Device:':<20} {device}")
        logger.debug(f"{'Pos Weight:':<20} {self.pos_weight:.4f}")
        logger.debug(f"{'-' * 80}")

    def _calculate_pos_weight(self, data, mask):
        """Calculate weight for positive class to handle class imbalance"""
        all_labels = []
        for i in range(len(data)):
            if mask[i]:
                for edge_type in data[i].y_dict:
                    all_labels.append(data[i].y_dict[edge_type])
        all_labels = torch.cat(all_labels)
        neg_pos_ratio = (all_labels == 0).sum() / (all_labels == 1).sum()
        return neg_pos_ratio.clone().detach()

    def train(self, model, optimizer):
        """Train the model with improved monitoring and class balance handling"""
        best_val_f1 = None
        patience = 15
        no_improve_epochs = 0

        # Use Focal Loss with class balancing
        criterion = FocalLoss(gamma=2.0, pos_weight=self.pos_weight)
        criterion = criterion.to(self.device)

        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train_epoch(model, optimizer, criterion)

            # Validation
            val_metrics = self.evaluate(model, self.val_loader)

            # Update best model
            if best_val_f1 is None or val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                no_improve_epochs = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'pos_weight': self.pos_weight
                }, f'{self.model_dir}/best_model.pt')
            else:
                no_improve_epochs += 1

            # print epoch metrics in a clean tabular format
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            logger.info(f"{'-' * 80}")
            logger.info(f"{'Metric':<15} {'Training':<15} {'Validation':<15}")
            logger.info(f"{'-' * 80}")
            logger.info(f"{'Loss':<15} {train_metrics['loss']:<15.4f} {'-':<15}")
            logger.info(
                f"{'Accuracy':<15} {train_metrics['accuracy']:<15.4f} {val_metrics['accuracy']:<15.4f}")
            logger.info(
                f"{'F1':<15} {train_metrics['f1']:<15.4f} {val_metrics['f1']:<15.4f}")
            logger.info(
                f"{'Precision':<15} {train_metrics['precision']:<15.4f} {val_metrics['precision']:<15.4f}")
            logger.info(
                f"{'Recall':<15} {train_metrics['recall']:<15.4f} {val_metrics['recall']:<15.4f}")
            logger.info(
                f"{'AUC-ROC':<15} {train_metrics['auc_roc']:<15.4f} {val_metrics['auc_roc']:<15.4f}")
            logger.info(f"{'-' * 80}")

            # print improvement status
            if val_metrics['f1'] > best_val_f1:
                logger.info("✓ New best model saved!")

            if no_improve_epochs >= patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # Load best model and evaluate on test set
        try:
            checkpoint = torch.load(
                f'{self.model_dir}/best_model.pt', weights_only=False)
        except Exception as e:
            logger.warning(
                f"Warning: Could not load checkpoint with weights_only=False: {str(e)}")
            checkpoint = torch.load(
                f'{self.model_dir}/best_model.pt', weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = self.evaluate(model, self.test_loader)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Final Test Results")
        logger.info(f"{'-' * 80}")
        logger.info(f"{'Metric':<15} {'Score':<15}")
        logger.info(f"{'-' * 80}")
        logger.info(f"{'Accuracy':<15} {test_metrics['accuracy']:<15.4f}")
        logger.info(f"{'F1':<15} {test_metrics['f1']:<15.4f}")
        logger.info(f"{'Precision':<15} {test_metrics['precision']:<15.4f}")
        logger.info(f"{'Recall':<15} {test_metrics['recall']:<15.4f}")
        logger.info(f"{'AUC-ROC':<15} {test_metrics['auc_roc']:<15.4f}")
        logger.info(f"{'=' * 80}")

    def train_epoch(self, model, optimizer, criterion):
        """Train for one epoch with configurable sampling strategy"""
        model.train()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        num_batches = 0

        # Get all data first
        all_data = list(self.train_loader.dataset)

        if self.sampling_strategy == SamplingStrategy.NONE:
            # For NONE strategy, use the train_loader directly
            data_iterator = self.train_loader
            num_iterations = len(self.train_loader)
        else:
            # For DYNAMIC and BALANCED strategies, calculate iterations based on dataset size
            num_iterations = len(all_data) // self.batch_size
            if len(all_data) % self.batch_size > 0:
                num_iterations += 1  # Add one more batch for the remainder

        data_iterator = iter(self.train_loader)
        for batch_idx in range(num_iterations):
            if self.sampler is not None:
                # Sample batch indices using configured sampler
                batch_indices = self.sampler.sample_batch_indices()
                batch_data = [all_data[i] for i in batch_indices]

                # Create a batch directly
                loader = DataLoader(batch_data, batch_size=len(
                    batch_data), shuffle=False)
                batch = next(iter(loader))
            else:
                try:
                    # For NONE strategy, use batches from data_iterator
                    batch = next(data_iterator)
                    # Calculate batch indices based on position in dataset
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, len(all_data))
                    batch_indices = list(range(start_idx, end_idx))
                except StopIteration:
                    logger.warning("Data iterator exhausted, reinitializing.")
                    # If we've exhausted the iterator, reinitialize it
                    data_iterator = iter(self.train_loader)
                    batch = next(data_iterator)
                    # Recalculate indices for the new batch
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, len(all_data))
                    batch_indices = list(range(start_idx, end_idx))

            batch = batch.to(self.device)

            optimizer.zero_grad()

            # Backward pass
            logits = model(batch.x_dict, batch.edge_index_dict,
                           batch.edge_attr_dict)
            target = torch.cat([batch.y_dict[edge_type].float()
                                for edge_type in batch.edge_index_dict.keys()])

            # Ensure logits and target have compatible shapes
            if logits.dim() > 1 and logits.size(1) > 1:
                logits = logits[:, 1]  # Take the positive class logit
            else:
                logits = logits.squeeze(-1)  # Remove any extra dimensions

            # Compute loss and backprop
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update the dynamic sampler with loss information for hard example mining
            if self.sampling_strategy == SamplingStrategy.DYNAMIC and self.sampler is not None:
                individual_losses = []
                with torch.no_grad():
                    # Calculate per-sample losses for updating the sampler
                    for i, idx in enumerate(batch_indices):
                        # If the loss calculation can't be done per-sample, we approximate
                        # This is a simplified approach - in practice, you might need to compute
                        # the loss per sample more precisely
                        if i < len(logits) and i < len(target):
                            individual_loss = criterion(
                                logits[i:i + 1], target[i:i + 1])
                            individual_losses.append(individual_loss)

                if individual_losses:
                    # Update sampling weights for hard example mining
                    self.sampler.update_mining_weights(
                        batch_indices, torch.tensor(individual_losses))

            # Get predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred_labels = (probs > self.threshold).float()
                all_preds.append(pred_labels.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(target.cpu())

            total_loss += loss.item()
            num_batches += 1

            # print batch statistics periodically
            self._print_batch_stats(
                batch_idx, loss.item(), logits, probs, target, pred_labels)

        # Compute epoch metrics
        metrics = self._compute_metrics(all_preds, all_probs, all_targets)
        metrics['loss'] = total_loss / \
                          num_batches if num_batches > 0 else float('inf')
        return metrics

    def evaluate(self, model, loader):
        """Evaluate model with all metrics including AUC-ROC"""
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = model(
                    batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                target = torch.cat([batch.y_dict[edge_type].float()
                                    for edge_type in batch.edge_index_dict.keys()])

                # Ensure logits have the right shape
                if logits.dim() > 1 and logits.size(1) > 1:
                    logits = logits[:, 1]  # Take the positive class logit
                else:
                    logits = logits.squeeze(-1)  # Remove any extra dimensions

                probs = torch.sigmoid(logits)
                pred_labels = (probs > self.threshold).float()

                all_preds.append(pred_labels.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(target.cpu())

        # Calculate metrics
        return self._compute_metrics(all_preds, all_probs, all_targets)

    def _compute_metrics(self, all_preds, all_probs, all_targets):
        """Compute F1, precision, recall, accuracy and AUC-ROC"""
        if not all_preds or not all_targets:
            return {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0, 'auc_roc': 0}

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        probs = torch.cat(all_probs).nan_to_num().numpy()

        return {
            'f1': f1_score(targets, preds, zero_division=0),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'accuracy': (preds == targets).mean(),
            'auc_roc': roc_auc_score(targets, probs)
        }

    def _print_batch_stats(self, batch_idx, loss, logits, probs, target, pred_labels):
        """Print detailed batch statistics"""
        if batch_idx % 10 != 0:
            return
        target_dist = torch.bincount(target.long())
        pred_dist = torch.bincount(pred_labels.long())
        logger.debug(f"\r[Batch {batch_idx:3d}] Loss: {loss:.4f} | "
              f"Class dist - Target: {target_dist.tolist()} Pred: {pred_dist.tolist()}")

    def _create_loader(self, data, mask, batch_size, shuffle):
        """Create data loader with configurable sampling strategy"""
        filtered_data = []
        total_graphs = sum(mask)
        filtered_indices = []

        for i in range(len(data)):
            if mask[i]:
                has_edges = any(
                    len(edges[0]) > 0 for edges in data[i].edge_index_dict.values())
                if has_edges:
                    filtered_data.append(data[i])
                    filtered_indices.append(i)

        filtered_out = total_graphs - len(filtered_data)
        if filtered_out > 0:
            logger.debug(f"Filtered {filtered_out} empty graphs from dataset")

        if shuffle:  # Training loader
            if self.sampling_strategy == SamplingStrategy.DYNAMIC:
                self.sampler = DynamicBatchSampler(
                    filtered_data,
                    batch_size=batch_size,
                    hard_mining_ratio=self.hard_mining_ratio
                )
                # Initialize with all indices in easy examples to start
                # Only after first few batches will the hard examples be identified
                self.sampler.easy_indices = list(range(len(filtered_data)))
                self.sampler.hard_indices = []  # Start with no hard examples

                logger.debug(
                    f"Initialized DynamicBatchSampler with {len(self.sampler.easy_indices)} easy examples")
                # We don't need the DataLoader's shuffling since our custom sampler handles that
                return DataLoader(filtered_data, batch_size=batch_size, shuffle=False)
            elif self.sampling_strategy == SamplingStrategy.BALANCED:  # BALANCED
                self.sampler = BalancedBatchSampler(
                    filtered_data,
                    batch_size=batch_size,
                    edge_percentage=1.0
                )
                logger.debug(
                    f"Initialized BalancedBatchSampler with {len(self.sampler.pos_weighted_indices)} positive and {len(self.sampler.neg_weighted_indices)} negative examples")
                # We don't need the DataLoader's shuffling since our custom sampler handles that
                return DataLoader(filtered_data, batch_size=batch_size, shuffle=False)
            else:  # NONE
                self.sampler = None
                # For NONE strategy, use standard DataLoader with shuffle
                return DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
        else:  # Validation/Test loader
            return DataLoader(filtered_data, batch_size=batch_size, shuffle=False)
