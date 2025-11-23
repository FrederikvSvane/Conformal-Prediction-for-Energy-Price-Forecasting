import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class EnergyPriceDataset(Dataset):
    def __init__(self, features, targets, sequence_length):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return X, y
    
def load_and_preprocess_energy_data(csv_path: str = '../data/energy_data.csv') -> pd.DataFrame:
    """
    Load and init energy price dataset
    """
    df = pd.read_csv(csv_path)

    # Convert date_time to hour, day_nr, week_nr and year
    df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
    df['day_nr'] = df['date_time'].dt.dayofweek + 1
    df['week_nr'] = df['date_time'].dt.isocalendar().week.astype('int32')
    df['year'] = df['date_time'].dt.year

    # Drop unneeded columns
    df.drop("intraday1", axis=1, inplace=True)
    df.drop("intraday2", axis=1, inplace=True)
    df.drop("intraday3", axis=1, inplace=True)
    df.drop("date_time", axis=1, inplace=True)

    # Create offset spot price column (previous spot price)
    df['spot_lag1'] = df['spot'].shift(1)
    df = df.dropna()  # Remove the first row (doesn't have a previous spot price)

    return df