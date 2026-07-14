import shap
import torch
import numpy as np

def explain_predictions(model, X_background, X_test, feature_names=None):
    """
    Uses SHAP to explain the PyTorch model predictions.
    DeepExplainer is a good choice for PyTorch models.
    """
    print("Generating SHAP explanations...")
    
    model.eval()
    
    # SHAP DeepExplainer requires PyTorch tensors
    background_tensor = torch.tensor(X_background, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    explainer = shap.DeepExplainer(model, background_tensor)
    
    # Calculate SHAP values for the test sample
    shap_values = explainer.shap_values(test_tensor)
    
    print("SHAP base values computed. Visualizations can be generated manually by calling shap plots.")
    return shap_values, explainer
