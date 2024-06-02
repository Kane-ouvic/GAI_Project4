import torch
import torch.optim as optim
from models.ddpm import DDPM
from utils.data_loader import load_image
import argparse
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import numpy as np
import os

def train_ddpm(image_path, num_epochs=10000, target_size=(32, 32)): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image(image_path).to(device)
    img = Resize(target_size)(img)

    # Load the DIP model's output as the initial prior for DDPM
    initial_prior = torch.load('dip_output.pt').to(device)

    # Initialize and train DDPM model
    ddpm_model = DDPM(initial_prior=initial_prior, device=device).to(device)
    ddpm_optimizer = optim.Adam(ddpm_model.parameters(), lr=0.0001) 

    losses = []
    output_dir = 'ddpm_output_images'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        ddpm_optimizer.zero_grad()
        loss = ddpm_model(img)
        loss.backward()
        ddpm_optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"DDPM Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

        if epoch % 1000 == 0:
            # Generate sample from DDPM
            sample = torch.randn(1, 3, *target_size, device=device)
            generated_img = ddpm_model.generate(sample)
            generated_img = generated_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            generated_img = np.clip(generated_img, 0, 1)  # Clamp values to [0, 1] range
            plt.imshow(generated_img)
            plt.show()
            torch.cuda.empty_cache()

            # Save generated image
            plt.imsave(f'{output_dir}/ddpm_output_epoch_{epoch}.png', generated_img)

    # Plot the loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDPM Training Loss Curve')
    plt.savefig('ddpm_loss_curve.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPM model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    train_ddpm(args.image_path)
