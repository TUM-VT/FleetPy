import torch
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import f1_score
from data_processing.config import DataProcessingConfig
from typing import Optional


class Trainer:
    DEFAULT_EPOCHS = 200
    THRESHOLD = 0.5

    def __init__(self, data, device, masks, config: Optional[DataProcessingConfig] = None, batch_size=32, epochs=200):
        """
        Initialize the trainer
        """
        self.device = device
        self.config = config or DataProcessingConfig()
        self.model_dir = os.path.join(self.config.base_data_dir, self.config.models_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = 0.5  # Threshold for binary classification
        
        # Create data loaders
        self.train_loader = self._create_loader(data, masks[0], batch_size, shuffle=True)
        self.val_loader = self._create_loader(data, masks[1], batch_size, shuffle=False)
        self.test_loader = self._create_loader(data, masks[2], batch_size, shuffle=False)

    def train(self, model, criterion, optimizer):
        """
        Train the model
        """
        best_val_f1 = 0
        no_improve_epochs = 0
        
        # Initialize scheduler with optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.7,
            patience=7,
            min_lr=1e-6
        )
        
        for epoch in range(self.epochs):
            loss = self.train_epoch(model, optimizer, criterion)
            val_f1 = self.evaluate(model, self.val_loader)
            
            # Update learning rate based on validation F1
            scheduler.step(val_f1)
            
            # Print progress
            print(f'Epoch {epoch+1}/{self.epochs}:')
            print(f'  Loss: {loss:.4f}')
            print(f'  Validation F1: {val_f1:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve_epochs = 0
                # Save best model
                torch.save(model.state_dict(), f'{self.model_dir}/best_model.pt')
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= 15:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(f'{self.model_dir}/best_model.pt'))
        test_f1 = self.evaluate(model, self.test_loader)
        print(f'Final Test F1: {test_f1:.4f}')

    def train_epoch(self, model, optimizer, criterion):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()  # Clear gradients.

            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)  # Perform a single forward pass.
            
            # Get the target labels for all edges and convert to float
            target = torch.cat([batch.y_dict[edge_type].float() for edge_type in batch.edge_index_dict.keys()])
            
            # Skip batches with NaN values in target
            if torch.isnan(target).any():
                print(f"Warning: NaN values in target for batch {batch_idx}")
                continue
            
            # Take only the positive class probability for binary classification
            out = out[:, 1]
            
            # Compute the loss for all edges in this batch
            loss = criterion(out, target)

            # Check for nan loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in batch {batch_idx}")
                continue

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self, model, loader):
        model.eval()
        total_f1 = 0
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                
                # Get the target labels for all edges and convert to float
                target = torch.cat([batch.y_dict[edge_type].float() for edge_type in batch.edge_index_dict.keys()])
                
                # Take only the positive class probability
                pred = pred[:, 1]
                pred_labels = (pred > self.threshold).float()
                
                # Skip batches with NaN values
                if torch.isnan(target).any() or torch.isnan(pred_labels).any():
                    continue

                # Convert to numpy for sklearn
                target_np = target.cpu().numpy()
                pred_labels_np = pred_labels.cpu().numpy()
                # Compute F1 score for the batch
                batch_f1 = f1_score(target_np, pred_labels_np, zero_division=0)
                total_f1 += batch_f1
                num_batches += 1
        return total_f1 / num_batches if num_batches > 0 else 0

    def _create_loader(self, data, mask, batch_size, shuffle):
        # Filter out empty graphs
        filtered_data = []
        for i in range(len(data)):
            if mask[i]:
                # Check if the graph has any edges
                has_edges = any(len(edges[0]) > 0 for edges in data[i].edge_index_dict.values())
                if has_edges:
                    filtered_data.append(data[i])
        
        print(f"Filtered out {len([i for i in range(len(data)) if mask[i]]) - len(filtered_data)} empty graphs")
        return DataLoader(filtered_data, batch_size=batch_size, shuffle=shuffle)
