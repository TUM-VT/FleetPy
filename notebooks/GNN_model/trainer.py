import torch


class Trainer:
    DEFAULT_EPOCHS = 200
    THRESHOLD = 0.5

    def __init__(self, save_model_dir, loader, device, masks, epochs=DEFAULT_EPOCHS, threshold=THRESHOLD):
        self.epochs = epochs
        self.threshold = threshold
        self.save_model_dir = save_model_dir
        self.loader = loader
        self.device = device
        self.train_idx, self.val_idx, self.test_idx = masks

    def train(self, model, criterion, optimizer):
        best_val_acc = 0
        for epoch in range(self.epochs):
            loss = self.train_epoch(model, optimizer, criterion)
            val_acc = self.evaluate(model, self.val_idx)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'{self.save_model_dir}model_best.pth')
            test_acc = self.evaluate(model, self.test_idx)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            torch.save(model.state_dict(), f'{self.save_model_dir}model_last.pth')

    def train_epoch(self, model, optimizer, criterion):
        model.train()
        total_loss = 0
        for batch in self.loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()  # Clear gradients.

            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)  # Perform a single forward pass.
            # Compute the loss solely based on the training nodes.
            loss = criterion(out[self.train_idx].view(-1), batch.y_dict[self.train_idx])

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

            total_loss += loss.item()

        return total_loss

    def evaluate(self, model, mask):
        model.eval()
        total_acc = 0
        num_batches = len(self.loader)
        with torch.no_grad():
            for batch in self.loader:
                batch = batch.to(self.device)
                pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                pred_labels = (pred.view(-1) > self.threshold).float()
                mean_batch_acc = (pred_labels[mask] == batch.y_dict[mask]).float().mean()
                total_acc += mean_batch_acc.item()
        return total_acc / num_batches
