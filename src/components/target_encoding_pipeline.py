import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger  import logging
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split



@dataclass
class TargetEncodingConfig:
    TargetEncoder_obj_file_path = os.path.join('artifacts','Target_Encoded.pkl')

class TargetEncoding:
    def __init__(self):
        self.Target_encoder_config = TargetEncodingConfig()

    def kfold_target_encoding_split(self,raw_path, target_col):
        # Sample split
        X = raw_path.drop(columns=[target_col])
        y = raw_path[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Columns to encode
        categorical_cols = ['source_center', 'destination_center']

        # Initialize encoded DataFrame
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        # K-Fold Setup
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # For each categorical column
        for col in categorical_cols:
            encoded_col = pd.Series(index=X_train.index, dtype=np.float64)

            # Cross-validation encoding for training set
            for train_idx, val_idx in kf.split(X_train):
                encoder = TargetEncoder(cols=[col])
                encoder.fit(X_train.iloc[train_idx][col], y_train.iloc[train_idx])

                # Transform only the validation fold
                encoded_vals = encoder.transform(X_train.iloc[val_idx][[col]])
                encoded_col.iloc[val_idx] = encoded_vals[col].values

            # Store the final encoded column
            X_train_encoded[col + '_te'] = encoded_col

            # Final encoder trained on full training data
            encoder_final = TargetEncoder(cols=[col])
            encoder_final.fit(X_train[col], y_train)

            # Apply to test set
            X_test_encoded[col + '_te'] = encoder_final.transform(X_test[[col]])[col]

        # Drop original categorical columns:
        X_train_encoded.drop(columns=categorical_cols, inplace=True)
        X_test_encoded.drop(columns=categorical_cols, inplace=True)

        return X_train_encoded, X_test_encoded, y_train, y_test
    


    


