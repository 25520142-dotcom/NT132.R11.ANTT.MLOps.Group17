import pandas as pd
import numpy as np

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    df = pd.concat([train_data, test_data], ignore_index=True)
    return df