import torch


class Trainer:
    DEFAULT_EPOCHS = 200
    THRESHOLD = 0.5

    def __init__(self, epochs=DEFAULT_EPOCHS, threshold=THRESHOLD):
        self.epochs = epochs
        self.threshold = threshold

    def train(self, model, data, criterion, optimizer):
        for epoch in range(self.epochs):
            loss = self.train_epoch(model, data, optimizer, criterion)
            val_acc = 0  # TODO fix
            test_acc = self.evaluate(model, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    def train_epoch(self, model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index, data.edge_label_index)  # Perform a single forward pass.
        # Compute the loss solely based on the training nodes.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_label_index)
            pred_labels = (pred > 0.5).float()
            acc = (pred_labels == data.edge_labels).sum().item() / data.edge_labels.size(0)
        return acc
