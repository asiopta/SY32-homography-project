import cv2
import numpy as np

img = cv2.imread("./seq1/001.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define HSV ranges for each color (tune these to your specific circles)
color_ranges = {
    "red":    ([0, 120, 70],   [10, 255, 255]),   # red wraps around 0°
    "red2":   ([170, 120, 70], [180, 255, 255]),   # red upper range
    "green":  ([40, 70, 70],   [80, 255, 255]),
    "blue":   ([100, 150, 50], [140, 255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255]),
}


# a function that returns an image where only the circles of a specific color are visible, and the rest of the image is black.
def apply_color_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return cv2.bitwise_and(img, img, mask=mask)



# a main function that applies the color masks to the image and displays the results.
def main():
    for color, (lower, upper) in color_ranges.items():
        masked_img = apply_color_mask(hsv, lower, upper)
        cv2.imshow(f"{color} circles", masked_img)
        # save result in a new picture
        cv2.imwrite(f"{color}_circles.png", masked_img)
        
