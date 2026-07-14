import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from models.local_model import LocalModel, train_model
from data.dataset_loader import load_data
from data.preprocessing import preprocess_data, get_dataloaders
from utils.config import INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS

class DDoSClient(fl.client.NumPyClient):
    """
    Flower Client for local training and model evaluation.
    """
    def __init__(self, model, train_loader, test_loader, epochs, lr, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        
    def get_parameters(self, config):
        """Returns model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def set_parameters(self, parameters):
        """Sets the local model parameters to the given ones."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        """Trains the local model on local data."""
        print(f"Client training started...")
        self.set_parameters(parameters)
        
        # Train locally
        train_model(self.model, self.train_loader, epochs=self.epochs, lr=self.lr, device=self.device)
        
        # Return updated parameters and size of training set
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
        
    def evaluate(self, parameters, config):
        """Evaluates the local model on the local test set."""
        print(f"Client evaluation started...")
        self.set_parameters(parameters)
        
        self.model.to(self.device)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                
                loss += criterion(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        accuracy = correct / total
        avg_loss = loss / len(self.test_loader)
        
        return float(avg_loss), len(self.test_loader.dataset), {"accuracy": accuracy}

def main():
    """Start the Flower Client."""
    print("Loading data for Client...")
    # Simulating localized client data
    X, y = load_data(use_dummy=True)
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = LocalModel(INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, OUTPUT_DIM)
    
    client = DDoSClient(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=device)
    
    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()
