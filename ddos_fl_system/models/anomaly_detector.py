from sklearn.ensemble import IsolationForest
import numpy as np

class ZeroDayDetector:
    """
    Anomaly Detector using Isolation Forest for Zero-day attack detection.
    """
    def __init__(self, contamination=0.05, random_state=42):
        """
        Args:
            contamination (float): The proportion of outliers in the data set. 
                                  Used when fitting to define the threshold.
        """
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X):
        """
        Fits the Isolation Forest on standard (normal) or mixed traffic.
        """
        print("Training Zero-Day Anomaly Detector...")
        self.model.fit(X)
        return self
        
    def predict(self, X):
        """
        Predicts anomalies.
        Returns:
            Normal (known traffic) -> 1
            Anomaly (zero-day attack) -> -1
        """
        return self.model.predict(X)
        
    def predict_labels(self, X):
        """
        Predicts and maps to standard labels (0: Normal, 1: Attack/Anomaly).
        """
        preds = self.predict(X)
        # Convert -1 (anomaly) to 1 (attack) and 1 (normal) to 0 (normal)
        return np.where(preds == -1, 1, 0)
