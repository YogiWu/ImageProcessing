#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_url = './hw2/hw2_input/01.png'
img = Image.open(img_url)
img_data = np.array(img)
plt.imshow(img, cmap="gray")

#%%
def get_val(data, i, j):
    if (i < 0 or j < 0 or i >= data.shape[0] or j >= data.shape[1]):
        return 0
    return data[i, j]

def filter2d(data, filter_matrix):
    new_data = np.zeros(data.shape)

    filter_width, filter_height = filter_matrix.shape
    pad_width = int(filter_width/2)
    pad_height = int(filter_height/2)

    for y in range(-pad_height, new_data.shape[1]-pad_height):
        for x in range(-pad_width, new_data.shape[0]-pad_width):
            total = 0.0

            for h in range(filter_height):
                for w in range(filter_width):
                    total += get_val(data, x+w, y+h) * filter_matrix[w][h] 

            new_data[x+pad_width][y+pad_height] = total

    return new_data

#%%
from scipy.signal import correlate2d 

n = 3
filter_matrix = np.ones((n, n))/(n*n)

new_img_data1 = filter2d(img_data, filter_matrix)
new_img_data2 = correlate2d(img_data, filter_matrix, mode='same')

plt.imshow(new_img_data1, cmap="gray")
plt.show()
plt.imshow(new_img_data2, cmap="gray")
plt.show()

#%%
n = 7
filter_matrix = np.ones((n, n))/(n*n)

new_img_data1 = filter2d(img_data, filter_matrix)
new_img_data2 = correlate2d(img_data, filter_matrix, mode='same')

plt.imshow(new_img_data1, cmap="gray")
plt.show()
plt.imshow(new_img_data2, cmap="gray")
plt.show()

#%%
n = 11
filter_matrix = np.ones((n, n))/(n*n)

new_img_data1 = filter2d(img_data, filter_matrix)
new_img_data2 = correlate2d(img_data, filter_matrix, mode='same')

plt.imshow(new_img_data1, cmap="gray")
plt.show()
plt.imshow(new_img_data2, cmap="gray")
plt.show()

# %%
laplacian_filter = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
])

new_img_data = filter2d(img_data, laplacian_filter)
plt.imshow(new_img_data, cmap="gray")
plt.show()

# %%
def hight_boost_filter(data, filter_matrix, k):
    new_data = filter2d(data, filter_matrix)

    return data+k*(data-new_data)

n=3
filter_matrix = np.ones((n, n)) / (n*n)

plt.imshow(hight_boost_filter(img_data, filter_matrix, 0.5), cmap="gray")
plt.show()

plt.imshow(hight_boost_filter(img_data, filter_matrix, 1), cmap="gray")
plt.show()

plt.imshow(hight_boost_filter(img_data, filter_matrix, 1.5), cmap="gray")
plt.show()

plt.imshow(hight_boost_filter(img_data, filter_matrix, 5), cmap="gray")
plt.show()
#%%