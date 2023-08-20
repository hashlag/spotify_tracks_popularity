import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset
import pandas as pd

class SpotifySet(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.scaler = StandardScaler()
        df = pd.read_csv(filename)

        self.genre_encoder = LabelEncoder()
        self.key_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        self.time_sign_encoder = LabelEncoder()

        df["genre"] = self.genre_encoder.fit_transform(df["genre"])
        df["key"] = self.key_encoder.fit_transform(df["key"])
        df["mode"] = self.mode_encoder.fit_transform(df["mode"])
        df["time_signature"] = self.time_sign_encoder.fit_transform(df["time_signature"])

        df.drop(["artist_name", "track_name", "track_id"], axis=1, inplace=True)

        self.X, self.y = [], []

        for _, row in df.iterrows():
            self.X.append(torch.tensor(list([row[0]]) + list(row[2:]), dtype=torch.float32))
            self.y.append(torch.tensor([row[1]], dtype=torch.float32))

        self.len = len(self.y)

        self.nX = self.scaler.fit_transform(self.X)

    def __getitem__(self, index):
        return self.nX[index], self.y[index]
    
    def __len__(self):
        return self.len