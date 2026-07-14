import pandas as pd
from sklearn.datasets import make_classification

def load_data(filepath=None, use_dummy=True):
    """
    Loads dataset from a CSV file or generates dummy data.
    
    Args:
        filepath (str): Path to CSV file.
        use_dummy (bool): If True, generates a synthetic dataset.
        
    Returns:
        pd.DataFrame, pd.Series: Features (X) and Labels (y)
    """
    if use_dummy or not filepath:
        print("Generating dummy dataset for DDoS detection...")
        # Simulating DDoS dataset: 20 features, 2 classes
        X, y = make_classification(
            n_samples=10000, 
            n_features=20, 
            n_classes=2, 
            weights=[0.8, 0.2], # 80% Normal, 20% DDoS
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y, name="label")
        return X_df, y_series
    else:
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        X_df = df.iloc[:, :-1]
        y_series = df.iloc[:, -1]
        return X_df, y_series
