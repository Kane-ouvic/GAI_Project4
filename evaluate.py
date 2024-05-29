import argparse
import torch
from models.ddpm import DDPM
from utils.data_loader import get_cifar10_dataloader
import matplotlib.pyplot as plt

def evaluate(ddpm_model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_cifar10_dataloader(data_path=data_path, train=False, batch_size=2, limit=100)

    ddpm = DDPM().to(device)
    ddpm.load_state_dict(torch.load(ddpm_model_path))

    for batch in dataloader:
        images, _ = batch
        images = images.to(device)

        generated_image = ddpm.generate(images.shape)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(images[0].cpu().permute(1, 2, 0))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(generated_image[0].cpu().permute(1, 2, 0))
        axes[1].set_title("Generated Image")
        axes[1].axis("off")

        plt.show()
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DDPM model")
    parser.add_argument("--ddpm_model_path", type=str, required=True, help="Path to the DDPM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CIFAR-10 dataset")
    args = parser.parse_args()

    evaluate(args.ddpm_model_path, args.data_path)
