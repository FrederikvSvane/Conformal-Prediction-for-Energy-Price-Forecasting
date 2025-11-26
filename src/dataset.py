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
    
def load_and_preprocess_energy_data(csv_path: str = '../data/energy_data.csv', keep_date: bool = False) -> pd.DataFrame:
    """
    Load and init energy price dataset

    Args:
        csv_path: Path to the CSV file
        keep_date: If True, preserve 'date_time' column for fold creation in CV
    """
    df = pd.read_csv(csv_path)

    # Convert date_time to hour, day_nr, week_nr and year
    df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
    df['day_nr'] = df['date_time'].dt.dayofweek + 1
    df['week_nr'] = df['date_time'].dt.isocalendar().week.astype('int32')
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month

    # Cyclical day-of-year encoding (smooth annual seasonality)
    day_of_year = df['date_time'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    # Heating demand interaction: uses "heating degree days" concept
    # When temp < 18°C, heating is needed. The colder it is, the more heating demand.
    # max(0, 18 - temp) gives heating intensity (0 in summer, high in winter)
    # Multiply by consumption to capture: cold weather + high demand = price pressure
    heating_demand = np.maximum(0, 18 - df['temp_forecast_dah_celcius'])
    df['heating_demand_interaction'] = heating_demand * df['consumption_forecast_dah_mw']

    # Temperature deviation from normal (hypothesis: unusual weather drives price spikes)
    df['temp_deviation'] = df['temp_forecast_dah_celcius'] - df['temp_norm_celcius']

    # Drop unneeded columns
    df.drop("intraday1", axis=1, inplace=True)
    df.drop("intraday2", axis=1, inplace=True)
    df.drop("intraday3", axis=1, inplace=True)
    if not keep_date:
        df.drop("date_time", axis=1, inplace=True)

    # Create offset spot price column (previous spot price)
    df['spot_lag1'] = df['spot'].shift(1)
    df = df.dropna()  # Remove the first row (doesn't have a previous spot price)

    return df