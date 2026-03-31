import numpy as np
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def detect_specific_borders(image_path):
    # 1. Load and Grayscale
    img = skimage.io.imread(image_path)
    if img.shape[-1] == 4: # Handle RGBA
        img = img[:, :, :3]
    gray = skimage.color.rgb2gray(img)
    
    # 2. Define Directional Kernels
    # These are designed to trigger a positive response for specific transitions
    kernels = {
        'Left Border (Black to White)':  np.array([[-1, 0, 1],
                                                   [-1, 0, 1],
                                                   [-1, 0, 1]]),
        
        'Right Border (White to Black)': np.array([[1, 0, -1],
                                                   [1, 0, -1],
                                                   [1, 0, -1]]),
        
        'Top Border (Black to White)':    np.array([[-1, -1, -1],
                                                   [ 0,  0,  0],
                                                   [ 1,  1,  1]]),
        
        'Bottom Border (White to Black)': np.array([[ 1,  1,  1],
                                                   [ 0,  0,  0],
                                                   [-1, -1, -1]])
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (name, kernel) in enumerate(kernels.items()):
        # Apply convolution
        edge_map = convolve(gray, kernel)
        
        # We only want the POSITIVE response (the specific transition)
        # Values < 0 are the opposite transition; we clip them to 0
        edge_map = np.clip(edge_map, 0, None)
        
        axes[i].imshow(edge_map, cmap='hot')
        axes[i].set_title(name)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Run the function
detect_specific_borders("Figure_1.png")