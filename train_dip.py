import torch
import torch.optim as optim
from models.dip import DeepImagePrior
from utils.data_loader import load_image
import matplotlib.pyplot as plt
import argparse
from torchvision.transforms import Resize
import os
import numpy as np

def train_dip(image_path, num_epochs=10000, target_size=(64, 64)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image(image_path).to(device)
    img = Resize(target_size)(img)  # Adjust the size of the input image
    model = DeepImagePrior().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    losses = []
    output_dir = 'dip_output_images'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(torch.randn(1, 3, *target_size, device=device))
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

        if epoch % 1000 == 0:
            output_img = output.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            output_img = np.clip(output_img, 0, 1)  # Clamp values to [0, 1] range
            plt.imsave(f'{output_dir}/output_img_epoch_{epoch}.png', output_img)
    
    # Save the DIP model's output for use in DDPM training
    torch.save(output, 'dip_output.pt')

    # Plot the loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('dip_loss_curve.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DIP model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    train_dip(args.image_path)
