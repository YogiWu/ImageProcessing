#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_data = np.array([
    [1, 2, 2, 1],
    [2, 3, 3, 2],
    [2, 3, 3, 2],
    [1, 2, 2, 1]
])

plt.imshow(img_data, cmap='gray')

# %%
n = 3

filter_array = np.ones((n, n))/(n*n)

# %%
from scipy.signal import correlate2d 

new_img_data = correlate2d(img_data, filter_array, mode='same')

print(new_img_data)
plt.imshow(new_img_data, cmap='gray')

# %%
