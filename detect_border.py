import numpy as np
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def detect_specific_borders(img, direction: str):
    '''
    direction =
         - 'L' for left
         - 'R' for right
         - 'T' for top
         - 'B' for bottom
    '''
    if direction not in ['L', 'R', 'T', 'B']:
        raise KeyError("value of direction need to be either: 'L', 'R', 'T' or 'B' ")

    #retrieve image (white_paper)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    gray = skimage.color.rgb2gray(img)
    
    # These directional kernels are designed to trigger a positive response for specific transitions
    # we got the idea from one of the videos of the youtube channel: 3Blue1Brown
    kernels = {
        'R': np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]]),
        
        'L': np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]]),
        
        'B': np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]]),
        
        'T': np.array([[ 1,  1,  1],
                       [ 0,  0,  0],
                       [-1, -1, -1]])
    }

    edge_map = convolve(gray, kernels[direction])
    edge_map = np.clip(edge_map, 0, None)


    edge_uint8 = (edge_map / edge_map.max() * 255).astype(np.uint8)
    return edge_uint8
    

import numpy as np
import skimage.io
import skimage.transform
import skimage.feature
import matplotlib.pyplot as plt

def get_edge_vector(image_path: str):
    '''
    Given an edge-detected image (output of detect_specific_borders),
    returns the dominant edge as a vector: (x1, y1) -> (x2, y2)
    '''
    edge_map = skimage.io.imread(image_path, as_gray=True)

    # Threshold to get a binary edge map
    binary = edge_map > (edge_map.max() * 0.5)

    # Hough Line Transform to find the dominant line
    hough_space, angles, dists = skimage.transform.hough_line(binary)

    # Extract the single most prominent line
    _, peak_angles, peak_dists = skimage.transform.hough_line_peaks(
        hough_space, angles, dists, num_peaks=1
    )

    angle = peak_angles[0]
    dist  = peak_dists[0]

    # Convert (angle, dist) to two endpoints spanning the image
    h, w = edge_map.shape
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Line equation: x*cos(a) + y*sin(a) = dist
    # Solve for endpoints on image borders
    if abs(sin_a) > abs(cos_a):          # more horizontal → use x = 0 and x = w
        x1, x2 = 0, w - 1
        y1 = (dist - x1 * cos_a) / sin_a
        y2 = (dist - x2 * cos_a) / sin_a
    else:                                 # more vertical → use y = 0 and y = h
        y1, y2 = 0, h - 1
        x1 = (dist - y1 * sin_a) / cos_a
        x2 = (dist - y2 * sin_a) / cos_a

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(edge_map, cmap='gray')
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label=f"Edge vector")
    ax.plot([x1, x2], [y1, y2], 'go', markersize=8)  # endpoints in green
    ax.legend()
    ax.set_title(f"Detected edge: ({x1},{y1}) → ({x2},{y2})")
    plt.tight_layout()
    plt.show()

    print(f"Edge vector: ({x1}, {y1}) → ({x2}, {y2})")
    return (x1, y1), (x2, y2)





if __name__ == "__main__":
    image_path = "white_paper.png"
    img = skimage.io.imread(image_path)
    
    '''
    # Run the function
    direction = "B"
    result = detect_specific_borders(img , direction)
    skimage.io.imsave(f"page_edges_{direction}.png", result)
    '''

    # Example usage
    p1, p2 = get_edge_vector("page_edges_B.png")
    l1, l2 = get_edge_vector("page_edges_L.png")



