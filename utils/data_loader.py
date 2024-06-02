from skimage import io
import torch
from torchvision.transforms import ToTensor, Resize

def load_image(image_path):
    image = io.imread(image_path)
    image = ToTensor()(image).unsqueeze(0)  # shape: (1, 3, H, W)
    return image
