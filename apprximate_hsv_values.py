import numpy as np
from skimage import io
import skimage
import matplotlib.pyplot as plt

base_image = io.imread("./seq1/001.jpg")

plt.figure()
plt.imshow(base_image)
plt.title('Click 4 points on the image')

# Select 4 points using ginput
# A plot window should appear. Click on the image to select 4 points.
# The program will wait until you have clicked 4 times.
coinsT = plt.ginput(4)

# Convert to numpy array for easier manipulation if needed
coinsT = np.array(coinsT)

print("Selected points:")
print(coinsT)

plt.show()