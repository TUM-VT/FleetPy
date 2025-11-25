import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from gnn_project.config import Config
from typing import Optional
import logging
from gnn_project.training.focal_loss import FocalLoss
from gnn_project.defaults import *

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, data, masks, optimizer, config: Config,
                 pos_weight: Optional[float] = None) -> None:
        """
        Initialize the trainer with configurable sampling strategy

        Args:
            data: Input data
            masks: Train/val/test masks
            optimizer: Optimizer for training
            config: Configuration object
            pos_weight: Optional weight for positive class
        """
        self.optimizer = optimizer
        self.config = config
        self.device = self.config.device
        self.epochs = self.config.epochs
        self.batch_size = self.config.batch_size
        self.threshold = self.config.classification_threshold
        self.model_dir = self.config.trained_models_dir

        # Create data loaders
        self.train_loader = self._create_loader(
            data, masks[TRAIN_MASKS], self.batch_size, shuffle=True)
        self.val_loader = self._create_loader(
            data, masks[VAL_MASKS], self.batch_size, shuffle=False)
        self.test_loader = self._create_loader(
            data, masks[TEST_MASKS], self.batch_size, shuffle=False)

        # Calculate class weights from training data
        if pos_weight is None:
            self.pos_weight = self._calculate_pos_weight(data, masks[TRAIN_MASKS])
        else:
            self.pos_weight = torch.tensor(pos_weight, device=self.device)

    def _calculate_pos_weight(self, data: list, mask: torch.Tensor) -> torch.Tensor:
        """Calculate weight for positive class to handle class imbalance

        Args:
            data: Input data
            mask: Mask indicating training samples

        Returns:
            Tensor representing the positive class weight
        """
        all_labels = []
        for i in range(len(data)):
            if mask[i]:
                for edge_type in data[i].y_dict:
                    all_labels.append(data[i].y_dict[edge_type])
        all_labels = torch.cat(all_labels)
        neg_pos_ratio = (all_labels == 0).sum() / (all_labels == 1).sum()
        return neg_pos_ratio.clone().detach()

    def train(self, model: torch.nn.Module) -> None:
        """Train the model

        Args:
            model: The GNN model to be trained
        """
        best_val_f1 = None
        patience = self.config.patience
        no_improve_epochs = 0

        # Use Focal Loss with class balancing
        criterion = FocalLoss(
            gamma=self.config.gamma, alpha=self.config.alpha, pos_weight=self.pos_weight)
        criterion = criterion.to(self.device)

        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train_epoch(model, criterion)

            # Validation
            val_metrics = self.evaluate(model, self.val_loader)

            # Update best model
            if best_val_f1 is None or val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                no_improve_epochs = 0
                # Save best model
                torch.save({
                    MODEL_STATE_DICT: model.state_dict(),
                    POS_WEIGHT: self.pos_weight
                }, self.config.saved_model_path)
            else:
                no_improve_epochs += 1

            # print epoch metrics in a clean tabular format
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            logger.info(f"{'-' * 80}")
            self.report_epoch_statistics(train_metrics, val_metrics)

            if no_improve_epochs >= patience:
                logger.info(
                    f'Early stopping triggered after {epoch + 1} epochs')
                break

        # Load best model and evaluate on test set
        try:
            checkpoint = torch.load(
                self.config.saved_model_path, weights_only=False)
        except Exception as e:
            logger.warning(
                f"Warning: Could not load checkpoint with weights_only=False: {str(e)}")
            checkpoint = torch.load(
                self.config.saved_model_path, weights_only=True)

        model.load_state_dict(checkpoint[MODEL_STATE_DICT])
        test_metrics = self.evaluate(model, self.test_loader)
        self.report_test_metrics(test_metrics)

    def report_test_metrics(self, test_metrics):
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

    def report_epoch_statistics(self, train_metrics, val_metrics):
        logger.info(f"{'Metric':<15} {'Training':<15} {'Validation':<15}")
        logger.info(f"{'-' * 80}")
        logger.info(
                f"{'Loss':<15} {train_metrics['loss']:<15.4f} {'-':<15}")
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

    def train_epoch(self, model: torch.nn.Module, criterion: torch.nn.Module) -> dict:
        """Train for one epoch

        Args:
            model: The GNN model to be trained
            criterion: Loss function

        Returns:
            Dictionary of training metrics
        """
        model.train()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

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
            self.optimizer.step()

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

    def evaluate(self, model: torch.nn.Module, loader: DataLoader) -> dict:
        """Evaluate model with all metrics

        Args:
            model: The GNN model to be evaluated
            loader: DataLoader for evaluation data

        Returns:
            Dictionary of evaluation metrics
        """
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

    def _compute_metrics(
        self,
        all_preds: list,
        all_probs: list,
        all_targets: list
    ) -> dict:
        """Compute F1, precision, recall, accuracy and AUC-ROC

        Args:
            all_preds: List of predicted labels
            all_probs: List of predicted probabilities
            all_targets: List of true labels

        Returns:
            Dictionary of computed metrics
        """
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

    def _print_batch_stats(
        self,
        batch_idx: int,
        loss: float,
        logits: torch.Tensor,
        probs: torch.Tensor,
        target: torch.Tensor,
        pred_labels: torch.Tensor
    ) -> None:
        """Print detailed batch statistics

        Args:
            batch_idx: Index of the current batch
            loss: Loss value for the batch
            logits: Logits from the model
            probs: Predicted probabilities
            target: True labels
            pred_labels: Predicted labels
        """
        if batch_idx % self.config.print_interval != 0:
            return
        target_dist = torch.bincount(target.long())
        pred_dist = torch.bincount(pred_labels.long())
        logger.debug(f"\r[Batch {batch_idx:3d}] Loss: {loss:.4f} | "
                     f"Class dist - Target: {target_dist.tolist()} Pred: {pred_dist.tolist()}")

    def _create_loader(
        self,
        data: list,
        mask: torch.Tensor,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Create data loader for training/validation/test

        Args:
            data: Input data
            mask: Mask indicating which samples to include
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data

        Returns:
            Configured DataLoader
        """
        filtered_data = []
        total_graphs = sum(mask)
        for i in range(len(data)):
            if mask[i].item():
                has_edges = any(
                    len(edges[0]) > 0 for edges in data[i].edge_index_dict.values())
                if has_edges:
                    filtered_data.append(data[i])
        filtered_out = total_graphs - len(filtered_data)
        if filtered_out > 0:
            logger.debug(f"Filtered {filtered_out} empty graphs from dataset")
        return DataLoader(filtered_data, batch_size=batch_size, shuffle=shuffle)
