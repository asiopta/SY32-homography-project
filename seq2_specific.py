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
    H_TOL = 0.05
    S_TOL = 0.15
    V_TOL = 0.15

    #manually selected hsv values of 4 different random points of the paper
    hsv_values = [
        (0.3167, 0.0429, 0.9137),
        (0.3519, 0.0388, 0.9098),
        (0.4048, 0.0314, 0.8745),
        (0.5909, 0.0474, 0.9098),
        [2.22222222e-01, 1.68539326e-02, 1.92987987e-17]
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
    selem = morphology.disk(15)
    #white_paper_mask = skimage.morphology.closing(combined_mask, selem)
    white_paper_mask = skimage.morphology.dilation(combined_mask, selem)

    #white_paper_mask = combined_mask

    # Fill any remaining holes completely
    #white_paper_mask = morphology.remove_small_holes(white_paper_mask, max_size=50000)

    # Remove small noisy blobs outside the paper
    #white_paper_mask = morphology.remove_small_objects(white_paper_mask, max_size=5000)

    white_paper_masked = img.copy()
    white_paper_masked[~white_paper_mask] = 0

    '''
    plt.figure()
    plt.imshow(white_paper_masked)
    plt.title("Detected paper area")
    plt.show()
    '''
    return white_paper_masked


def detect_red(img):
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
        'r': (0.9931, 0.6621, 0.5686)
    }

    (h, s, v) = hsv_values.get('r', "error")

    # Tolerance for each channel
    H_TOL = 0.04
    S_TOL = 0.2
    V_TOL = 0.2

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

    '''
    plt.figure()
    plt.imshow(img)
    plt.title(f"Mask for color red")
    plt.show()
    '''
    return img


'''
def assign_coordinates_to_corners(coords_circles_centers, dict_coords_prev_image):
    
    takes as input:
      -the result of coord_circle_center(): array of 4 (x,y) coordinates
      -and the dict of the previous image with the corner as key and the coordinate as value

    idea:
        if dict_coords_prev_image is empty ( the values of each corner are (None, None) ), 
        we can determin the order of the corners in the first image by looking at the relative position of the 4 detected coordinates,
        for example the one with the smallest x and y will be the top left corner, the one with the smallest x and biggest y will be the bottom left corner, etc...
        
        if dict_coords_prev_image is not empty, we can use the previous coordinates to assign the corners in the current image by looking at the distance between the detected coordinates and the previous coordinates, 
        for example the coordinate that is closest to the previous top left corner will be assigned to the top left corner in the current image, etc...

        result:
        a dictionary with the corner as key and the coordinate as value, for example:
        res = {
            'TL': (tl_x, tl_y) ,
            'TR': (tr_x, tr_y) ,
            'BR': (br_x, br_y) ,
            'BL': (bl_x, bl_y)
        }
    
    

    res = {}

    return res

'''

def coord_circle_center(masked_image, min_area=100):
    '''
    given a mask containing only the wanted color/area,
    return the centers of the 4 largest regions
    '''
    non_black_mask = np.any(masked_image != 0, axis=2)
    
    labeled = skimage.measure.label(non_black_mask)
    if labeled.max() == 0:
        return []
    
    region_sizes = np.bincount(labeled.flat)[1:]  # skip background (0)
    
    # Get up to 4 largest regions above min_area threshold
    valid_labels = np.where(region_sizes >= min_area)[0] + 1
    if len(valid_labels) == 0:
        return []
    
    # Sort by size descending, take top 4
    valid_labels = sorted(valid_labels, key=lambda l: region_sizes[l-1], reverse=True)[:4]

    results = []
    for label in valid_labels:
        region = (labeled == label)
        rows, cols = np.where(region)
        center = (np.mean(cols), np.mean(rows))
        results.append(center)

    return results


def assign_coordinates_to_corners(coords_circles_centers, dict_coords_prev_image):
    '''
    takes as input:
      - the result of coord_circle_center(): array of 4 (x,y) coordinates
      - and the dict of the previous image with the corner as key and the coordinate as value
    '''

    res = {}

    if len(coords_circles_centers) == 0:
        return {'TL': (None, None), 'TR': (None, None), 'BR': (None, None), 'BL': (None, None)}

    coords = np.array(coords_circles_centers)  # shape (N, 2), each row is (x, y)

    all_none = all(v == (None, None) for v in dict_coords_prev_image.values())

    if all_none:
        # --- First frame: assign by relative position ---
        # TL = smallest x+y, TR = smallest y but largest x, etc.
        scores = {
            'TL': coords[:, 0] + coords[:, 1],          # min x+y
            'TR': -coords[:, 0] + coords[:, 1],         # min -x+y  (large x, small y)
            'BR': -(coords[:, 0] + coords[:, 1]),        # max x+y
            'BL': coords[:, 0] - coords[:, 1],           # min x-y  (small x, large y)
        }

        assigned = set()
        for corner, score in scores.items():
            # Pick the best unassigned index
            sorted_indices = np.argsort(score)
            for idx in sorted_indices:
                if idx not in assigned:
                    res[corner] = (coords[idx, 0], coords[idx, 1])
                    assigned.add(idx)
                    break

    else:
        # --- Subsequent frames: assign by proximity to previous coordinates ---
        assigned_coords = set()
        for corner, prev_coord in dict_coords_prev_image.items():
            if prev_coord == (None, None):
                res[corner] = (None, None)
                continue

            prev = np.array(prev_coord)
            distances = np.linalg.norm(coords - prev, axis=1)

            # Pick closest unassigned coordinate
            sorted_indices = np.argsort(distances)
            for idx in sorted_indices:
                if idx not in assigned_coords:
                    res[corner] = (coords[idx, 0], coords[idx, 1])
                    assigned_coords.add(idx)
                    break

    return res


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
    #tform = AffineTransform()
    #tform.estimate(src, dst)
    tform = AffineTransform.from_estimate(src, dst)

    # Step 4: apply to the 4th point from the previous image
    p4 = np.array([dict_coords_prev_image[missing_color]], dtype=float)
    p4_transformed = tform(p4)

    last_point_coord = (p4_transformed[0][0], p4_transformed[0][1])
    dict_coord_curr_image[missing_color] = last_point_coord

    #print(f"Predicted missing coordinate for color '{missing_color}': {last_point_coord}")
    print(f"Updated dict_coord_curr_image: {dict_coord_curr_image}")

    return dict_coord_curr_image




def apply_homography_single_image(fennec, img_path, dict_coords_prev_image = {}):

    img_base = skimage.io.imread(img_path)
    #img_base = undistort(img_base)

    #detect whiter paper
    white_paper_mask = detect_inside_paper(img_base)

    #detect red circles and the coordinates of the centers
    red_mask = detect_red(white_paper_mask)
    coords_circles_centers = coord_circle_center(red_mask) #array of 4 (x,y) coordinates
    dict_coord_curr_image = assign_coordinates_to_corners(coords_circles_centers, dict_coords_prev_image)
    print(dict_coord_curr_image)

    '''
    # if we fail a detect a color, we deduce it from the other 3 and the previous coordinates
    for color, coords in dict_coord_curr_image.items():
        if coords == (None, None):
            dict_coord_curr_image = predict_missing_coordinate(dict_coord_curr_image, dict_coords_prev_image)
            break
    '''
    # coins
    coinsI = np.array([[0, 0], [WIDHT_FENNEC, 0], [WIDHT_FENNEC, HEIGHT_FENNEC], [0, HEIGHT_FENNEC]])
    # order:            TL              TR                      BR                        BL

    coinsO = np.array(list(dict_coord_curr_image.values()))
    # order:  TL, TR, BR, BL  

    tform = skimage.transform.estimate_transform('projective', coinsI, coinsO)
    H = tform.params

    fennec_homographie = transform(fennec, H, hw = img_base.shape[:2], interp='linear')

    result = img_base.copy()
    mask = (fennec_homographie[:, :, 0] != 0)
    result[mask] = fennec_homographie[mask]

    #save the result of the homography for each image
    '''
    plt.figure()
    plt.imshow(result)
    plt.show()
    ''' 
    result_path = os.path.join("results", img_path.replace(".png", "_result.png"))
    skimage.io.imsave(result_path, result)

    return dict_coord_curr_image


if __name__ == "__main__":
    
    dict_coords_prev_image =  {
        'TL': (None, None),
        'TR': (None, None),
        'BR': (None, None),
        'BL': (None, None)
    }
    

    #import images and define constants
    img_paths = get_png_files("./seq2")

    fennec = skimage.io.imread("fennec.jpg")
    HEIGHT_FENNEC, WIDHT_FENNEC = fennec.shape[:2]
    coinsI = np.array([[0, 0], [WIDHT_FENNEC, 0], [WIDHT_FENNEC, HEIGHT_FENNEC], [0, HEIGHT_FENNEC]])

    for img_path in img_paths:
        # apply only starting from 126th image
        '''
        first_image_number = 140
        image_number = int(os.path.basename(img_path).split(".")[0])
        if image_number < first_image_number:
            continue
        '''
        print(f"Processing {img_path}...")
        
        dict_coord_curr_image = apply_homography_single_image(fennec, img_path, dict_coords_prev_image)
            
        dict_coords_prev_image = dict_coord_curr_image

