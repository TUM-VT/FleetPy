import torch
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from data_processing.config import DataProcessingConfig
from typing import Optional


class Trainer:
    def __init__(self, data, device, masks, config: Optional[DataProcessingConfig] = None, batch_size=32, epochs=200):
        """Initialize the trainer with improved handling of class imbalance"""
        self.device = device
        self.config = config or DataProcessingConfig()
        self.model_dir = os.path.join(self.config.base_data_dir, self.config.models_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = 0.5  # Fixed threshold for binary classification
        
        # Create data loaders
        self.train_loader = self._create_loader(data, masks[0], batch_size, shuffle=False)
        self.val_loader = self._create_loader(data, masks[1], batch_size, shuffle=False)
        self.test_loader = self._create_loader(data, masks[2], batch_size, shuffle=False)
        
        # Calculate class weights from training data
        self.pos_weight = self._calculate_pos_weight(data, masks[0])
        print(f"\n{'='*80}")
        print(f"Training Configuration")
        print(f"{'-'*80}")
        print(f"{'Batch Size:':<20} {batch_size}")
        print(f"{'Max Epochs:':<20} {epochs}")
        print(f"{'Device:':<20} {device}")
        print(f"{'Pos Weight:':<20} {self.pos_weight:.4f}")
        print(f"{'-'*80}")

    def _calculate_pos_weight(self, data, mask):
        """Calculate weight for positive class to handle class imbalance"""
        all_labels = []
        for i in range(len(data)):
            if mask[i]:
                for edge_type in data[i].y_dict:
                    all_labels.append(data[i].y_dict[edge_type])
        all_labels = torch.cat(all_labels)
        neg_pos_ratio = (all_labels == 0).sum() / (all_labels == 1).sum()
        return torch.tensor(neg_pos_ratio, device=self.device)

    def train(self, model, optimizer):
        """Train the model with improved monitoring and class balance handling"""
        best_val_f1 = 0
        patience = 15
        no_improve_epochs = 0
        
        # Use BCEWithLogitsLoss with pos_weight
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
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
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"{'-'*80}")
            print(f"{'Metric':<15} {'Training':<15} {'Validation':<15}")
            print(f"{'-'*80}")
            print(f"{'Loss':<15} {train_metrics['loss']:<15.4f} {'-':<15}")
            print(f"{'Accuracy':<15} {train_metrics['accuracy']:<15.4f} {val_metrics['accuracy']:<15.4f}")
            print(f"{'F1':<15} {train_metrics['f1']:<15.4f} {val_metrics['f1']:<15.4f}")
            print(f"{'Precision':<15} {train_metrics['precision']:<15.4f} {val_metrics['precision']:<15.4f}")
            print(f"{'Recall':<15} {train_metrics['recall']:<15.4f} {val_metrics['recall']:<15.4f}")
            print(f"{'AUC-ROC':<15} {train_metrics['auc_roc']:<15.4f} {val_metrics['auc_roc']:<15.4f}")
            print(f"{'-'*80}")
            
            # Print improvement status
            if val_metrics['f1'] > best_val_f1:
                print("✓ New best model saved!")
            
            if no_improve_epochs >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load best model and evaluate on test set
        try:
            checkpoint = torch.load(f'{self.model_dir}/best_model.pt', weights_only=False)
        except Exception as e:
            print(f"Warning: Could not load checkpoint with weights_only=False: {str(e)}")
            checkpoint = torch.load(f'{self.model_dir}/best_model.pt', weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = self.evaluate(model, self.test_loader)
        print(f"\n{'='*80}")
        print(f"Final Test Results")
        print(f"{'-'*80}")
        print(f"{'Metric':<15} {'Score':<15}")
        print(f"{'-'*80}")
        print(f"{'Accuracy':<15} {test_metrics['accuracy']:<15.4f}")
        print(f"{'F1':<15} {test_metrics['f1']:<15.4f}")
        print(f"{'Precision':<15} {test_metrics['precision']:<15.4f}")
        print(f"{'Recall':<15} {test_metrics['recall']:<15.4f}")
        print(f"{'AUC-ROC':<15} {test_metrics['auc_roc']:<15.4f}")
        print(f"{'='*80}")

    def train_epoch(self, model, optimizer, criterion):
        """Train for one epoch with improved monitoring"""
        model.train()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            target = torch.cat([batch.y_dict[edge_type].float() for edge_type in batch.edge_index_dict.keys()])
            
            # Ensure logits and target have compatible shapes
            if logits.dim() > 1 and logits.size(1) > 1:
                logits = logits[:, 1]  # Take the positive class logit
            else:
                logits = logits.squeeze(-1)  # Remove any extra dimensions
            print('Logits:', logits[:10], 'Target:', target[:10])  # Debugging line

            # Compute loss and backprop
            loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            
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
            # if batch_idx % 10 == 0:
            self._print_batch_stats(batch_idx, loss.item(), logits, probs, target, pred_labels)
        
        # Compute epoch metrics
        metrics = self._compute_metrics(all_preds, all_probs, all_targets)
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
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
                logits = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                target = torch.cat([batch.y_dict[edge_type].float() for edge_type in batch.edge_index_dict.keys()])
                
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
              f"Class dist - Target: {target_dist.tolist()} Pred: {pred_dist.tolist()}", 
              end="", flush=True)

    def _create_loader(self, data, mask, batch_size, shuffle):
        """Create data loader with empty graph filtering"""
        filtered_data = []
        total_graphs = sum(mask)
        for i in range(len(data)):
            if mask[i]:
                has_edges = any(len(edges[0]) > 0 for edges in data[i].edge_index_dict.values())
                if has_edges:
                    filtered_data.append(data[i])
        
        filtered_out = total_graphs - len(filtered_data)
        if filtered_out > 0:
            print(f"Filtered {filtered_out} empty graphs from dataset")
        return DataLoader(filtered_data, batch_size=batch_size, shuffle=shuffle)
