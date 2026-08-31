from typing import Tuple, List, Any
import numpy as np
from sklearn.model_selection import KFold

def train_cv(model_cls: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> Tuple[List[Any], np.ndarray]:
    """Train a model using K-Fold cross validation and return models + out-of-fold predictions."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = model_cls()
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)
        
    return models, oof_preds
