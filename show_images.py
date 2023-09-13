import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Specify the path to the saved image file
image_path = './logged_files/CIFAR10/balmy-disco-3/images_0.pt'

# Load the image tensor from the file
image_tensor = torch.load(image_path)

# Create a grid of images
grid_tensor = vutils.make_grid(image_tensor, nrow=10, ncol=10, padding=2, normalize=True)  # Adjust nrow as needed

# Convert the grid tensor to a PIL Image for visualization
to_pil = transforms.ToPILImage()
grid_image = to_pil(grid_tensor)

# Display the grid of images using matplotlib
plt.figure(figsize=(12, 12))  # Adjust the figure size as needed
plt.imshow(grid_image)
plt.axis('off')  # Hide the axis values
plt.savefig("images.png")
plt.show()
