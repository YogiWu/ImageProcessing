#%%
### hw 2.2 Scaling

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
img_url = './hw1/hw1_input/00.png'
img = mpimg.imread(img_url)

print(img.shape, img[0][0])

### Origin Image
plt.imshow(img, cmap='gray')

#%%
from PIL import Image

img2 = Image.open(img_url)
img_data = np.array(img2)

print(img_data.shape, img_data[0][0])
plt.imshow(img_data, cmap='gray')

#%%
def scale(img_data, width, height):
    new_img = Image.new(img_data.mode, img_data.size, 0)
    new_img_data = np.array(new_img)

    img_width = img_data.shape[0]
    img_height = img_data.shape[1]

    

#%%
scale_list = np.array([
    [192, 128],
    [96, 64],
    [48, 32],
    [24, 16],
    [12, 8],
    [300, 200],
    [450, 300],
    [500, 200]
])
    



# %%
