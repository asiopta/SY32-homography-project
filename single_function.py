import numpy as np
import skimage
import matplotlib.pyplot as plt
import os

def get_png_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.endswith(".png")
    ]

from skimage import morphology
import numpy as np
import skimage

def detect_inside_paper(img):
    img_hsv = skimage.color.rgb2hsv(img[:, :, :3])

    H_TOL = 0.2
    S_TOL = 0.02
    V_TOL = 0.1

    hsv_values = [
        (0.3167, 0.0429, 0.9137),
        (0.3519, 0.0388, 0.9098),
        (0.4048, 0.0314, 0.8745),
        (0.5909, 0.0474, 0.9098)
    ]

    # Build a mask for each HSV point and combine them with OR
    combined_mask = np.zeros(img_hsv.shape[:2], dtype=bool)
    for (h, s, v) in hsv_values:
        s_mask = (img_hsv[:, :, 1] >= s - S_TOL) & (img_hsv[:, :, 1] <= s + S_TOL)
        v_mask = (img_hsv[:, :, 2] >= v - V_TOL) & (img_hsv[:, :, 2] <= v + V_TOL)

        h_low, h_high = h - H_TOL, h + H_TOL
        if h_low < 0:
            h_mask = (img_hsv[:, :, 0] >= h_low + 1) | (img_hsv[:, :, 0] <= h_high)
        elif h_high > 1:
            h_mask = (img_hsv[:, :, 0] >= h_low) | (img_hsv[:, :, 0] <= h_high - 1)
        else:
            h_mask = (img_hsv[:, :, 0] >= h_low) & (img_hsv[:, :, 0] <= h_high)

        combined_mask |= h_mask & s_mask & v_mask

    # Morphological closing to fill holes inside the paper
    # disk size controls how aggressively gaps are filled — increase if needed
    selem = morphology.disk(50)
    white_paper_mask = skimage.morphology.closing(combined_mask, selem)

    # Fill any remaining holes completely
    white_paper_mask = morphology.remove_small_holes(white_paper_mask, max_size=50000)

    # Remove small noisy blobs outside the paper
    white_paper_mask = morphology.remove_small_objects(white_paper_mask, max_size=5000)

    white_paper_masked = img.copy()
    white_paper_masked[~white_paper_mask] = 0
    return white_paper_masked


def detect_color(img, color: str):

    #retrieve image
    img_hsv = skimage.color.rgb2hsv(img)

    # get the hsv values of the color we wish to detect
    hsv_values = {
        'y': (0.0966, 0.5245, 0.8000), #yellow
        'r': (0.9931, 0.6621, 0.5686), #red
        'g': (0.5333, 0.2083, 0.1882), #green
        'b': (0.6712, 0.4205, 0.3451) #blue
    }

    (h, s, v) = hsv_values.get(color, "error")

    # Tolerance for each channel
    H_TOL = 0.01
    S_TOL = 0.15
    V_TOL = 0.15

    # Build per-channel masks
    s_mask = (img_hsv[:, :, 1] >= s - S_TOL) & (img_hsv[:, :, 1] <= s + S_TOL)
    v_mask = (img_hsv[:, :, 2] >= v - V_TOL) & (img_hsv[:, :, 2] <= v + V_TOL)

    h_low, h_high = h - H_TOL, h + H_TOL
    if h_low < 0:
        h_mask = (img_hsv[:, :, 0] >= h_low + 1) | (img_hsv[:, :, 0] <= h_high)
    elif h_high > 1:
        h_mask = (img_hsv[:, :, 0] >= h_low) | (img_hsv[:, :, 0] <= h_high - 1)
    else:
        h_mask = (img_hsv[:, :, 0] >= h_low) & (img_hsv[:, :, 0] <= h_high)

    combined_mask = h_mask & s_mask & v_mask

    selem = morphology.disk(50)
    combined_mask = skimage.morphology.closing(combined_mask, selem)

    # Apply mask: keep original pixels where mask is True, else black
    img[~combined_mask] = 0
        
    return img


def coord_circle_center(masked_image):
    non_black_mask = np.any(masked_image != 0, axis=2)
    
    # Keep only the largest connected region to ignore noise
    labeled = skimage.measure.label(non_black_mask)
    if labeled.max() == 0:
        return None
    
    # Find the largest region
    region_sizes = np.bincount(labeled.flat)[1:]  # skip background (0)
    largest_label = np.argmax(region_sizes) + 1
    largest_region = (labeled == largest_label)
    
    rows, cols = np.where(largest_region)
    return (np.mean(cols), np.mean(rows))


if __name__ == "__main__":
    img_paths = get_png_files("./seq1")

    #for img in img_paths:
    img = img_paths[190]
    img_test = skimage.io.imread(img)

    white_paper_mask = detect_inside_paper(img_test)

    
    #yellow
    yellow_mask = detect_color(white_paper_mask.copy() , 'y')
    (yx, yy) = coord_circle_center(yellow_mask)

    #red
    red_mask = detect_color(white_paper_mask.copy() , 'r')
    (rx, ry) = coord_circle_center(red_mask)

    #green
    green_mask = detect_color(white_paper_mask.copy() , 'g')
    (gx, gy) = coord_circle_center(green_mask)

    #blue
    blue_mask = detect_color(white_paper_mask.copy() , 'b')
    (bx, by) = coord_circle_center(blue_mask)
    
    

    plt.plot()
    plt.imshow(white_paper_mask)

    plt.scatter(yx, yy, color='black', marker='x', s=30)
    plt.scatter(rx, ry, color='black', marker='x', s=30)
    plt.scatter(gx, gy, color='black', marker='x', s=30)
    plt.scatter(bx, by, color='black', marker='x', s=30)
    

    plt.show()



