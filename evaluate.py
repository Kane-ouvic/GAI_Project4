import torch
from models.dip import DeepImagePrior
from models.ddpm import DDPM
from utils.data_loader import load_image
import matplotlib.pyplot as plt
import argparse
from torchvision.transforms import Resize
import numpy as np

def evaluate(image_path, target_size=(32, 32)):  # 调整图像大小以适应模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image(image_path).to(device)
    img = Resize(target_size)(img)

    # Load and train DIP model to get initial prior
    dip_model = DeepImagePrior().to(device)
    dip_optimizer = torch.optim.Adam(dip_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Train DIP model for a number of epochs to get the initial prior
    for epoch in range(500):  # Number of epochs for DIP training
        dip_optimizer.zero_grad()
        dip_output = dip_model(torch.randn(1, 3, *target_size, device=device))
        dip_loss = criterion(dip_output, img)
        dip_loss.backward()
        dip_optimizer.step()

    # Use DIP output as initial prior for DDPM
    initial_prior = dip_output.detach()

    # Initialize and train DDPM model
    ddpm_model = DDPM(initial_prior, device=device).to(device)
    ddpm_optimizer = torch.optim.Adam(ddpm_model.parameters(), lr=0.0001)

    for epoch in range(10000):  # Number of epochs for DDPM training
        ddpm_optimizer.zero_grad()
        loss = ddpm_model(img)
        loss.backward()
        ddpm_optimizer.step()

        if epoch % 100 == 0:
            print(f"DDPM Epoch [{epoch}/10000], Loss: {loss.item():.4f}")

    # Generate sample from DDPM
    sample = torch.randn(1, 3, *target_size, device=device)
    generated_img = ddpm_model.generate(sample)

    # Convert the generated image to numpy and clamp values to [0, 1]
    generated_img = generated_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    generated_img = np.clip(generated_img, 0, 1)

    # Display generated image
    plt.imshow(generated_img)
    plt.show()

    # Save generated image
    plt.imsave('generated_image.png', generated_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DDPM model with DIP initialization')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    evaluate(args.image_path)
