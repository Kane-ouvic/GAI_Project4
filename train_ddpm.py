import argparse
import torch
from models.ddpm import DDPM
from models.dip import DIP
from utils.data_loader import get_cifar10_dataloader

def train_ddpm(dip_model_path, output_path, data_path, num_steps=1000, lr=0.001, data_limit=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_cifar10_dataloader(data_path=data_path, train=True, batch_size=16, limit=data_limit)

    dip_model = DIP().to(device)
    dip_model.load_state_dict(torch.load(dip_model_path))

    ddpm = DDPM().to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for step in range(num_steps):
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)
            initial_prior = dip_model(images).detach()

            ddpm.initialize_with_prior(initial_prior)

            optimizer.zero_grad()
            output = ddpm.step()
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()

            # 清理缓存
            torch.cuda.empty_cache()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    torch.save(ddpm.state_dict(), output_path)
    print(f"DDPM model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM model with DIP prior")
    parser.add_argument("--dip_model_path", type=str, required=True, help="Path to the DIP model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the DDPM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CIFAR-10 dataset")
    args = parser.parse_args()

    train_ddpm(args.dip_model_path, args.output_path, args.data_path)
