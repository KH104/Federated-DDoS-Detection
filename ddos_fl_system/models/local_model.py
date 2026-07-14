import torch
import torch.nn as nn

class LocalModel(nn.Module):
    """
    Feedforward Neural Network for DDoS attack detection (Known attacks).
    Architecture: Input -> Dense -> ReLU -> Dense -> ReLU -> Output
    """
    def __init__(self, input_dim, hidden_dim_1=64, hidden_dim_2=32, output_dim=2):
        super(LocalModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, epochs, lr, device="cpu"):
    """
    Trains the local PyTorch model on the dataset.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
    return model

def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluates the model on the test set and computes accuracy and loss.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            loss += criterion(outputs, batch_y).item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    accuracy = correct / total
    avg_loss = loss / len(test_loader)
    
    return avg_loss, accuracy
