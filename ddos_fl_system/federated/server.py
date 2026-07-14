import flwr as fl
from utils.config import FL_ROUNDS, MIN_AVAILABLE_CLIENTS, FRACTION_FIT

def main():
    """Start the Flower Server."""
    print("Starting Federated Server...")
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FRACTION_FIT, # Sample 100% of available clients for training
        fraction_evaluate=1.0, # Sample 100% of available clients for evaluation
        min_fit_clients=MIN_AVAILABLE_CLIENTS, # Never sample less than this number of clients for training
        min_evaluate_clients=MIN_AVAILABLE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
    )
    
    # Start server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
