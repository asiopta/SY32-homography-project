import numpy as np
import skimage
from skimage import morphology
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata


def get_png_files(folder_path):
    '''
    returns a list containing the path of each image in a given folder
    '''
    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.endswith(".png")
    ]

def detect_inside_paper(img):
    '''
    given an image, it detects a white paper and everything inside
    return an image where only the paper is visible, evrything else black
    '''
    img_hsv = skimage.color.rgb2hsv(img[:, :, :3])

    #error rate allowed per variable
    # could be fine tuned further
    H_TOL = 0.2
    S_TOL = 0.02
    V_TOL = 0.1

    #manually selected hsv values of 4 different random points of the paper
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
    # disk size controls how aggressively gaps are filled, increase if needed
    selem = morphology.disk(50)
    white_paper_mask = skimage.morphology.closing(combined_mask, selem)

    # Fill any remaining holes completely
    #white_paper_mask = morphology.remove_small_holes(white_paper_mask, max_size=50000)

    # Remove small noisy blobs outside the paper
    #white_paper_mask = morphology.remove_small_objects(white_paper_mask, max_size=5000)

    white_paper_masked = img.copy()
    white_paper_masked[~white_paper_mask] = 0
    return white_paper_masked


def detect_color(img, color: str):
    '''
    given an image and the first letter of one of these colors:
    yellow, red, green, blue

    It detects the color in the image and return a mask containing only the specified color
    '''
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
    '''
    given a mask containing only the wanted color/area,
    return the center 
    '''
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
    return np.mean(cols), np.mean(rows)


def transformer(I, H, hw = (-1,-1), interp='linear'):
  h = hw[0]
  w = hw[1]
  if (w <= 0 or h <= 0):
    (h,w) = hw = I.shape[:2]
  O = np.zeros((h,w)) # image de sortie
  xx1, yy1 = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0]))
  xx1 = xx1.flatten()
  yy1 = yy1.flatten()
  Hinv = np.linalg.inv(H)
  xx2, yy2 = np.meshgrid(np.arange(O.shape[1]), np.arange(O.shape[0]))
  xx2 = xx2.flatten()
  yy2 = yy2.flatten()
  xxyy2 = np.stack((xx2,yy2,np.ones((O.size))), axis=0)
  xxyy = Hinv @ xxyy2
  xxyy = np.stack((xxyy[0]/xxyy[2], xxyy[1]/xxyy[2]), axis=0)
  O = griddata((xx1,yy1), I.flatten(), xxyy.T, method=interp, fill_value=0).reshape(O.shape)
  return O



if __name__ == "__main__":
    img_paths = get_png_files("./seq1")

    fennec = skimage.io.imread("fennec.jpg")
    #for img in img_paths:
    img = img_paths[190]
    img_test = skimage.io.imread(img)

    white_paper_mask = detect_inside_paper(img_test)

    
    #yellow
    yellow_mask = detect_color(white_paper_mask.copy() , 'y')
    yx, yy = coord_circle_center(yellow_mask)

    #red
    red_mask = detect_color(white_paper_mask.copy() , 'r')
    rx, ry = coord_circle_center(red_mask)

    #green
    green_mask = detect_color(white_paper_mask.copy() , 'g')
    gx, gy = coord_circle_center(green_mask)

    #blue
    blue_mask = detect_color(white_paper_mask.copy() , 'b')
    bx, by = coord_circle_center(blue_mask)
    

    # coins
     
    height, width = fennec.shape[:2]

    coinsI = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    coinsO = np.array([[yx, yy],
                     [rx, ry],
                     [gx, gy],
                     [bx, by]]
    )

    tform = skimage.transform.estimate_transform('projective', coinsI, coinsO)
    H = tform.params

    fennec_homographie = transformer(fennec, H, hw = img_test.shape[:2], interp='linear')

    result = img_test.copy()
    mask = (fennec_homographie[:, :, 0] != 0)
    result[mask] = fennec_homographie[mask]

    plt.figure()
    plt.imshow(result)
    plt.legend()
    plt.show()


    '''
    plt.figure()
    plt.imshow(fennec_homographie, cmap='gray')
    plt.scatter(coinsO[:, 0], coinsO[:, 1], color='red', marker='o', s=50, label='Selected Points')
    plt.legend()
    plt.show()

    plt.plot()
    plt.imshow(white_paper_mask)

    plt.scatter(yx, yy, color='black', marker='x', s=30)
    plt.scatter(rx, ry, color='black', marker='x', s=30)
    plt.scatter(gx, gy, color='black', marker='x', s=30)
    plt.scatter(bx, by, color='black', marker='x', s=30)
    

    plt.show()
    '''



