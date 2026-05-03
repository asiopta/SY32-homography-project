import numpy as np
from skimage import io
import skimage
import matplotlib.pyplot as plt

'''
base_image = io.imread("./results/seq4/139_result.png")

plt.figure()
plt.imshow(base_image)
plt.title('Click 4 points on the image')

# Select 4 points using ginput
# A plot window should appear. Click on the image to select 4 points.
# The program will wait until you have clicked 4 times.
coinsT = plt.ginput(1)

# Convert to numpy array for easier manipulation if needed
coinsT = np.array(coinsT)

print("Selected points:")
print(coinsT)

plt.show()

points = [[367.03246753, 381.64285714],
 [409.24025974, 380.01948052],
 [373.52597403, 412.48701299],
 [412.48701299, 407.61688312],
 [519.62987013, 401.12337662]]

H_TOL = 0.2
S_TOL = 0.02
V_TOL = 0.1

hsv_values = [
    (0.3167, 0.0429, 0.9137),
    (0.3519, 0.0388, 0.9098),
    (0.4048, 0.0314, 0.8745),
    (0.5909, 0.0474, 0.9098)
]
'''
rgb_values = [[139, 128, 123],
              [72, 60, 65]]

# transform rgb to hsv
hsv_values = [skimage.color.rgb2hsv(np.array([[color]])) for color in rgb_values]
print("HSV values:", hsv_values)

