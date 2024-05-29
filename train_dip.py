import argparse
import torch
from models.dip import DIP
from utils.data_loader import get_cifar10_dataloader

def train_dip(output_path, data_path, num_epochs=1000, lr=0.01, data_limit=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_cifar10_dataloader(data_path=data_path, train=True, batch_size=16, limit=data_limit)

    model = DIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), output_path)
    print(f"DIP model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DIP model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the DIP model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CIFAR-10 dataset")
    args = parser.parse_args()

    train_dip(args.output_path, args.data_path)
