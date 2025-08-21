import torch
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from data_processing.config import DataProcessingConfig
from typing import Optional, List, Dict
from enum import Enum
import random


class SamplingStrategy(Enum):
    DYNAMIC = "dynamic"  # Uses hard example mining
    BALANCED = "balanced"  # Uses class-balanced sampling


class BalancedBatchSampler:
    """Samples batches with balanced class distribution."""
    def __init__(self, data: List, batch_size: int, balance_ratio: float = 1.0):
        self.data = data
        self.batch_size = batch_size
        self.balance_ratio = balance_ratio
        self.pos_indices, self.neg_indices = self._split_by_class()
        
    def _split_by_class(self) -> tuple[list, list]:
        """Split indices by class label."""
        pos_indices, neg_indices = [], []
        for idx, graph in enumerate(self.data):
            # Get all labels from the graph
            all_labels = []
            for edge_type in graph.y_dict:
                all_labels.extend(graph.y_dict[edge_type].tolist())
                
            # TODO fix: If graph has any positive labels, consider it positive
            if any(label == 1 for label in all_labels):
                pos_indices.append(idx)
            else:
                neg_indices.append(idx)
        return pos_indices, neg_indices
    
    def sample_batch_indices(self) -> list:
        """Sample a balanced batch based on the balance ratio."""
        pos_samples_count = int(self.batch_size / (1 + self.balance_ratio))
        neg_samples_count = self.batch_size - pos_samples_count
        
        pos_samples = random.sample(self.pos_indices, min(pos_samples_count, len(self.pos_indices)))
        neg_samples = random.sample(self.neg_indices, min(neg_samples_count, len(self.neg_indices)))
        
        # Combine and shuffle
        batch_indices = pos_samples + neg_samples
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
        # Get probabilities
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        
        # Calculate focal weights
        focal_weight = targets * (1 - probs).pow(self.gamma) + (1 - targets) * probs.pow(self.gamma)
        
        # Calculate losses for positive and negative classes
        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        
        # Apply class balancing
        if self.pos_weight is not None:
            alpha_weight = targets * self.pos_weight + (1 - targets)
        else:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            
        # Combine all terms
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss.mean()


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
        sorted_indices = sorted(self.loss_history.items(), key=lambda x: x[1], reverse=True)
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
                    size=min(missing, len(self.hard_indices) - len(hard_samples)),
                    replace=False
                )
                additional_samples.extend(additional_hard)
                missing -= len(additional_hard)
                
            if missing > 0 and len(self.easy_indices or []) > len(easy_samples):
                # Sample more from easy if needed and available
                additional_easy = np.random.choice(
                    [i for i in self.easy_indices if i not in easy_samples],
                    size=min(missing, len(self.easy_indices) - len(easy_samples)),
                    replace=False
                )
                additional_samples.extend(additional_easy)
                
        # Combine all samples
        batch_indices = list(hard_samples) + list(easy_samples) + list(additional_samples)
        random.shuffle(batch_indices)
        return batch_indices


class Trainer:
    def __init__(self, data, device, masks, config: Optional[DataProcessingConfig] = None, batch_size=32, epochs=200,
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
        self.config = config or DataProcessingConfig()
        self.model_dir = os.path.join(
            self.config.base_data_dir, self.config.models_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = 0.5  # Fixed threshold for binary classification
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

        print(f"\n{'=' * 80}")
        print(f"Training Configuration")
        print(f"{'-' * 80}")
        print(f"{'Batch Size:':<20} {batch_size}")
        print(f"{'Max Epochs:':<20} {epochs}")
        print(f"{'Device:':<20} {device}")
        print(f"{'Pos Weight:':<20} {self.pos_weight:.4f}")
        print(f"{'-' * 80}")

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
        best_val_f1 = 0
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
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                no_improve_epochs = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'pos_weight': self.pos_weight
                }, f'{self.model_dir}/best_model.pt')
            else:
                no_improve_epochs += 1

            # Print epoch metrics in a clean tabular format
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'-' * 80}")
            print(f"{'Metric':<15} {'Training':<15} {'Validation':<15}")
            print(f"{'-' * 80}")
            print(f"{'Loss':<15} {train_metrics['loss']:<15.4f} {'-':<15}")
            print(
                f"{'Accuracy':<15} {train_metrics['accuracy']:<15.4f} {val_metrics['accuracy']:<15.4f}")
            print(
                f"{'F1':<15} {train_metrics['f1']:<15.4f} {val_metrics['f1']:<15.4f}")
            print(
                f"{'Precision':<15} {train_metrics['precision']:<15.4f} {val_metrics['precision']:<15.4f}")
            print(
                f"{'Recall':<15} {train_metrics['recall']:<15.4f} {val_metrics['recall']:<15.4f}")
            print(
                f"{'AUC-ROC':<15} {train_metrics['auc_roc']:<15.4f} {val_metrics['auc_roc']:<15.4f}")
            print(f"{'-' * 80}")

            # Print improvement status
            if val_metrics['f1'] > best_val_f1:
                print("✓ New best model saved!")

            if no_improve_epochs >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # Load best model and evaluate on test set
        try:
            checkpoint = torch.load(
                f'{self.model_dir}/best_model.pt', weights_only=False)
        except Exception as e:
            print(
                f"Warning: Could not load checkpoint with weights_only=False: {str(e)}")
            checkpoint = torch.load(
                f'{self.model_dir}/best_model.pt', weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = self.evaluate(model, self.test_loader)
        print(f"\n{'=' * 80}")
        print(f"Final Test Results")
        print(f"{'-' * 80}")
        print(f"{'Metric':<15} {'Score':<15}")
        print(f"{'-' * 80}")
        print(f"{'Accuracy':<15} {test_metrics['accuracy']:<15.4f}")
        print(f"{'F1':<15} {test_metrics['f1']:<15.4f}")
        print(f"{'Precision':<15} {test_metrics['precision']:<15.4f}")
        print(f"{'Recall':<15} {test_metrics['recall']:<15.4f}")
        print(f"{'AUC-ROC':<15} {test_metrics['auc_roc']:<15.4f}")
        print(f"{'=' * 80}")

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
        
        # Debug: Print learning rate
        # for param_group in optimizer.param_groups:
        #     print(f"[DEBUG] Learning rate: {param_group['lr']}")
        #     break
            
        # Number of iterations equivalent to full dataset coverage
        num_iterations = len(all_data) // self.batch_size
        if len(all_data) % self.batch_size > 0:
            num_iterations += 1  # Add one more batch for the remainder
            
        for batch_idx in range(num_iterations):
            # Sample batch indices using configured sampler
            batch_indices = self.sampler.sample_batch_indices()
            batch_data = [all_data[i] for i in batch_indices]
            
            # Create a batch directly
            batch = DataLoader(batch_data, batch_size=len(batch_data), shuffle=False).__iter__().next()
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
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
            optimizer.step()
            
            # Update the dynamic sampler with loss information for hard example mining
            if self.sampling_strategy == SamplingStrategy.DYNAMIC:
                individual_losses = []
                with torch.no_grad():
                    # Calculate per-sample losses for updating the sampler
                    for i, idx in enumerate(batch_indices):
                        # If the loss calculation can't be done per-sample, we approximate
                        # This is a simplified approach - in practice, you might need to compute
                        # the loss per sample more precisely
                        if i < len(logits) and i < len(target):
                            individual_loss = criterion(logits[i:i+1], target[i:i+1])
                            individual_losses.append(individual_loss)
                
                if individual_losses:
                    # Update sampling weights for hard example mining
                    self.sampler.update_mining_weights(batch_indices, torch.tensor(individual_losses))

            # Get predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred_labels = (probs > self.threshold).float()
                all_preds.append(pred_labels.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(target.cpu())

            total_loss += loss.item()
            num_batches += 1

            # Print batch statistics periodically
            self._print_batch_stats(
                batch_idx, loss.item(), logits, probs, target, pred_labels)

        # Compute epoch metrics
        print('preds', all_preds)
        print('probs', all_probs)
        print('targets', all_targets)
        metrics = self._compute_metrics(all_preds, all_probs, all_targets)
        metrics['loss'] = total_loss / \
                          num_batches if num_batches > 0 else float('inf')
        return metrics

    def evaluate(self, model, loader):
        """Evaluate model with all metrics including AUC-ROC"""
        model.eval()
        print(f"[DEBUG] Model in eval mode: {model.training}")  # Should be False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                print(f"[DEBUG] Dropout active? training={module.training}, p={module.p}")
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                print(f"[DEBUG] Val edge types: {list(batch.edge_index_dict.keys())}")
                print(f"[DEBUG] Val y_dict keys: {list(batch.y_dict.keys())}")
                for k, v in batch.edge_attr_dict.items():
                    print(f"[VAL EDGE ATTR] {k}: mean={v.mean():.4f}, std={v.std():.4f}, shape={v.shape}")
                logits = model(
                    batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                target = torch.cat([batch.y_dict[edge_type].float()
                                    for edge_type in batch.edge_index_dict.keys()])

                # Ensure logits have the right shape
                if logits.dim() > 1 and logits.size(1) > 1:
                    logits = logits[:, 1]  # Take the positive class logit
                else:
                    logits = logits.squeeze(-1)  # Remove any extra dimensions

                print(f"[VAL LOGITS] mean={logits.mean():.4f}, std={logits.std():.4f}, min={logits.min():.4f}, max={logits.max():.4f}")
                probs = torch.sigmoid(logits)
                print(f"[VAL PROBS] mean={probs.mean():.4f}, std={probs.std():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")
                # Optional: visualize probability distribution
                try:
                    import matplotlib.pyplot as plt
                    plt.hist(probs.cpu().numpy(), bins=50)
                    plt.title("Validation Probability Distribution")
                    plt.xlabel("Probability")
                    plt.ylabel("Frequency")
                    plt.show()
                except Exception as e:
                    print(f"[DEBUG] Could not plot histogram: {e}")
                pred_labels = (probs > self.threshold).float()

                all_preds.append(pred_labels.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(target.cpu())

        print('val preds', all_preds)
        print('val probs', all_probs)
        print('val targets', all_targets)
        # Calculate metrics
        return self._compute_metrics(all_preds, all_probs, all_targets)

    def _compute_metrics(self, all_preds, all_probs, all_targets):
        """Compute F1, precision, recall, accuracy and AUC-ROC"""
        if not all_preds or not all_targets:
            return {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0, 'auc_roc': 0}

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        probs = torch.cat(all_probs).numpy()

        return {
            'f1': f1_score(targets, preds, zero_division=0),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'accuracy': (preds == targets).mean(),
            'auc_roc': roc_auc_score(targets, probs)
        }

    def _print_batch_stats(self, batch_idx, loss, logits, probs, target, pred_labels):
        """Print detailed batch statistics"""
        target_dist = torch.bincount(target.long())
        pred_dist = torch.bincount(pred_labels.long())
        print(f"\r[Batch {batch_idx:3d}] Loss: {loss:.4f} | "
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
            print(f"Filtered {filtered_out} empty graphs from dataset")
            
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
                
                print(f"Initialized DynamicBatchSampler with {len(self.sampler.easy_indices)} easy examples")
                
            else:  # BALANCED
                self.sampler = BalancedBatchSampler(
                    filtered_data,
                    batch_size=batch_size,
                    balance_ratio=self.balance_ratio
                )
                print(f"Initialized BalancedBatchSampler with {len(self.sampler.pos_indices)} positive and {len(self.sampler.neg_indices)} negative examples")
                
            # We don't need the DataLoader's shuffling since our custom sampler handles that
            return DataLoader(filtered_data, batch_size=batch_size, shuffle=False)
        else:  # Validation/Test loader
            return DataLoader(filtered_data, batch_size=batch_size, shuffle=False)
