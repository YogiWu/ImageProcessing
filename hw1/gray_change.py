#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_url = './hw1/hw1_input/00.png'
img = Image.open(img_url)
plt.imshow(img, cmap="gray")

#%%
def quantize(img_data, gray_level):
    rate = (gray_level-1) / 255

    new_image_data = np.array(img_data, dtype=np.float)
    new_image_data *= rate

    new_image_data = np.round(new_image_data) / rate

    return np.round(new_image_data)

#%%
gray_level_list = [128, 32, 8, 4, 2]

for gray_level in gray_level_list:
    # plt.figure(figsize=(9,6))
    plt.imshow(quantize(np.array(img), gray_level), cmap='gray')
    plt.show()


# %%
import IPython.display

for gray_level in gray_level_list:
    IPython.display.display(Image.fromarray(np.uint8(quantize(np.array(img), gray_level)), 'L'))

# %%
