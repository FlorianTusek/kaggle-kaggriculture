import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe and generate features for training or inference."""
    df_feat = df.copy()
    # Add custom feature engineering transformations here
    return df_feat
