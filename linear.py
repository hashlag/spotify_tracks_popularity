import torch
import torch.nn as nn
import dataset
import models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def main():
    dst = dataset.SpotifySet("SpotifyFeatures.csv")

    train_set, test_set = random_split(dst, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=150, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=150, shuffle=False)

    model = models.LinearRegression()

    crt = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    train_loss_history = []

    for epoch in range(8):
        for step, (X, y) in enumerate(train_loader):
            pred = model(X.type(torch.float32))
            loss = crt(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step+1) % 100 == 0:
                print(f"epoch {epoch+1} | step {step+1} | loss {loss.item():.4f}")
                train_loss_history.append(loss.item())

    model.eval()

    with torch.no_grad():
        predictions, real = [], []
        for step, (X, y) in enumerate(test_loader):
            pred = model(X.type(torch.float32))
            predictions.extend(pred.flatten().tolist())
            real.extend(y.flatten().tolist())
        mae = sum([abs(p - r) for p, r in zip(predictions, real)]) / len(real)
        print(f"Test set MAE: {mae:.4f}")

    plt.plot(train_loss_history, label="train loss")
    plt.grid(True, axis="y")
    plt.xlabel("step x100")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()