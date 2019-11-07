#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_url = './hw2/hw2_input/00.png'
img = Image.open(img_url)
img_data = np.array(img)
plt.imshow(img, cmap="gray")


# %%
def histogram(data):
    y = np.zeros(256)

    for sub_array in data:
        for item in sub_array:
            y[item] += 1

    return y

def show_histogram(data):
    plt.bar(np.arange(256), histogram(data))
    plt.show()

show_histogram(img_data)

# %%
def equalize_hist(data):
    gray_historgram = histogram(data)

    t = np.arange(256)

    sum_list = np.cumsum(gray_historgram)
    for i in range(256):
        t[i] = 255 / data.shape[0] / data.shape[1] * sum_list[i]

    new_img_data = np.zeros(data.shape, dtype=np.int)

    for y in range(data.shape[1]):
        for x in range(data.shape[0]):
            new_img_data[x, y] = t[data[x, y]]

    return new_img_data


# %%
new_img_data1 = equalize_hist(img_data)

plt.imshow(new_img_data1, cmap='gray')
plt.show()
show_histogram(new_img_data1)

new_img_data2 = equalize_hist(new_img_data1)

plt.imshow(new_img_data2, cmap='gray')
plt.show()
show_histogram(new_img_data2)

# %%
