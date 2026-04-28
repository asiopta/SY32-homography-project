import numpy as np
import skimage
from skimage import morphology
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
from skimage.transform import AffineTransform


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

    plt.figure()
    plt.imshow(white_paper_masked)
    plt.title("Detected paper area")
    plt.show()

    return white_paper_masked


def detect_color(img, color: str):
    '''
    given an image and the first letter of one of these colors:
    yellow, red, green, blue

    It detects the color in the image and return a mask containing only the specified color
    '''
    #retrieve image
    img = img.copy()
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
    H_TOL = 0.03
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

    selem = morphology.disk(5)
    combined_mask = skimage.morphology.closing(combined_mask, selem)

    # Apply mask: keep original pixels where mask is True, else black
    img[~combined_mask] = 0

    plt.figure()
    plt.imshow(img)
    plt.title(f"Mask for color '{color}'")
    plt.show()
        
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
        return None, None
    
    # Find the largest region
    region_sizes = np.bincount(labeled.flat)[1:]  # skip background (0)
    largest_label = np.argmax(region_sizes) + 1
    largest_size = region_sizes[largest_label - 1]

    # Reject if the largest region is too small
    if largest_size < 200:  # threshold can be adjusted based on expected size of the color area
        return None, None

    largest_region = (labeled == largest_label)
    
    rows, cols = np.where(largest_region)
    return np.mean(cols), np.mean(rows)


def transform(I, H, hw=(-1, -1), interp='linear'):
    h, w = hw
    if (w <= 0 or h <= 0):
        h, w = I.shape[:2]
    
    # Initialize output image with the correct number of channels
    if I.ndim == 3:
        O = np.zeros((h, w, I.shape[2]))
    else:
        O = np.zeros((h, w))

    # Prep coordinates
    xx1, yy1 = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0]))
    xx1, yy1 = xx1.flatten(), yy1.flatten()
    
    Hinv = np.linalg.inv(H)
    xx2, yy2 = np.meshgrid(np.arange(w), np.arange(h))
    xx2, yy2 = xx2.flatten(), yy2.flatten()
    
    xxyy2 = np.stack((xx2, yy2, np.ones(xx2.size)), axis=0)
    xxyy = Hinv @ xxyy2
    xxyy = np.stack((xxyy[0]/xxyy[2], xxyy[1]/xxyy[2]), axis=0)

    # Apply interpolation channel by channel if RGB
    if I.ndim == 3:
        for i in range(I.shape[2]):
            channel = I[:, :, i].flatten()
            O[:, :, i] = griddata((xx1, yy1), channel, xxyy.T, method=interp, fill_value=0).reshape(h, w)
    else:
        O = griddata((xx1, yy1), I.flatten(), xxyy.T, method=interp, fill_value=0).reshape(h, w)
        
    return O



def predict_missing_coordinate(dict_coord_curr_image, dict_coords_prev_image):
    # Step 1: find the missing color (None, None) in current image
    missing_color = None
    for color, coords in dict_coord_curr_image.items():
        if coords == (None, None):
            missing_color = color
            break

    if missing_color is None:
        raise ValueError("No missing coordinate found in dict_coord_curr_image")
    
    if dict_coords_prev_image.get(missing_color) == (None, None):
        raise ValueError("Cannot predict missing coordinate because it is also missing in dict_coords_prev_image")

    # Step 2: build src/dst arrays from the 3 known matched points
    src_points = []
    dst_points = []

    for color, dst_coords in dict_coord_curr_image.items():
        if color == missing_color:
            continue
        src_coords = dict_coords_prev_image[color]
        src_points.append(src_coords)
        dst_points.append(dst_coords)

    src = np.array(src_points, dtype=float)
    dst = np.array(dst_points, dtype=float)

    # Step 3: estimate affine transform
    tform = AffineTransform.from_estimate(src, dst)

    # Step 4: apply to the 4th point from the previous image
    p4 = np.array([dict_coords_prev_image[missing_color]], dtype=float)
    p4_transformed = tform(p4)

    last_point_coord = (p4_transformed[0][0], p4_transformed[0][1])
    dict_coord_curr_image[missing_color] = last_point_coord

    #print(f"Predicted missing coordinate for color '{missing_color}': {last_point_coord}")
    print(f"Updated dict_coord_curr_image: {dict_coord_curr_image}")

    return dict_coord_curr_image




def apply_homography_single_image(fennec, img_base, dict_coords_prev_image = {}):
    #detect whiter paper
    white_paper_mask = detect_inside_paper(img_base)

    #detect colors and the coordinates of the center

    #yellow
    yellow_mask = detect_color(white_paper_mask, 'y')
    yx, yy = coord_circle_center(yellow_mask)

    #red
    red_mask = detect_color(white_paper_mask, 'r')
    rx, ry = coord_circle_center(red_mask)

    #green
    green_mask = detect_color(white_paper_mask, 'g')
    gx, gy = coord_circle_center(green_mask)

    #blue
    blue_mask = detect_color(white_paper_mask, 'b')
    bx, by = coord_circle_center(blue_mask)

    dict_coord_curr_image = {
        'y': (yx, yy) if yx is not None else (None, None),
        'r': (rx, ry) if rx is not None else (None, None),
        'g': (gx, gy) if gx is not None else (None, None),
        'b': (bx, by) if bx is not None else (None, None)
    }

    print(dict_coord_curr_image)

    # if we fail a detect a color, we deduce it from the other 3 and the previous coordinates
    for color, coords in dict_coord_curr_image.items():
        if coords == (None, None):
            dict_coord_curr_image = predict_missing_coordinate(dict_coord_curr_image, dict_coords_prev_image)
            break

    # coins
    coinsI = np.array([[0, 0], [WIDHT_FENNEC, 0], [WIDHT_FENNEC, HEIGHT_FENNEC], [0, HEIGHT_FENNEC]])
    # order:            TL              TR                      BR                        BL

    coinsO = np.array(list(dict_coord_curr_image.values()))
    # order:  y, r, g, b  ← whatever order the dict was defined in

    coinsO = np.array(list(dict_coord_curr_image.values()))
    '''
    coinsO = np.array([[yx, yy],
                     [rx, ry],
                     [gx, gy],
                     [bx, by]])
    '''

    tform = skimage.transform.estimate_transform('projective', coinsI, coinsO)
    H = tform.params

    fennec_homographie = transform(fennec, H, hw = img_base.shape[:2], interp='linear')

    result = img_base.copy()
    mask = (fennec_homographie[:, :, 0] != 0)
    result[mask] = fennec_homographie[mask]

    plt.figure()
    plt.imshow(result)
    plt.show()

    return dict_coord_curr_image


if __name__ == "__main__":

    dict_coords_prev_image =  {
        'y': (None, None),
        'r': (None, None),
        'g': (None, None),
        'b': (None, None)
    }
    #import images and define constants
    img_paths = get_png_files("./seq4b")

    fennec = skimage.io.imread("fennec.jpg")
    HEIGHT_FENNEC, WIDHT_FENNEC = fennec.shape[:2]
    coinsI = np.array([[0, 0], [WIDHT_FENNEC, 0], [WIDHT_FENNEC, HEIGHT_FENNEC], [0, HEIGHT_FENNEC]])

    for img in img_paths:
        print(f"Processing {img}...")
        img_base = skimage.io.imread(img)


        dict_coord_curr_image = apply_homography_single_image(fennec, img_base, dict_coords_prev_image)
            
        dict_coords_prev_image = dict_coord_curr_image

