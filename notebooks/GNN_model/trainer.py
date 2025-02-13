import torch


class Trainer:
    DEFAULT_EPOCHS = 200
    THRESHOLD = 0.5

    def __init__(self, save_model_dir, epochs=DEFAULT_EPOCHS, threshold=THRESHOLD):
        self.epochs = epochs
        self.threshold = threshold
        self.save_model_dir = save_model_dir

    def train(self, model, data, criterion, optimizer):
        best_val_acc = 0
        for epoch in range(self.epochs):
            loss = self.train_epoch(model, data, optimizer, criterion)
            val_acc = 0  # TODO fix
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'{self.save_model_dir}model_best.pth')
            test_acc = self.evaluate(model, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            torch.save(model.state_dict(), f'{self.save_model_dir}model_last.pth')

    def train_epoch(self, model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x_dict, data.edge_index_dict)  # Perform a single forward pass.
        # Compute the loss solely based on the training nodes.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            pred = model(data.x_dict, data.edge_index_dict)
            pred_labels = (pred > self.threshold).float()
            correct = (pred_labels[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
        return acc
