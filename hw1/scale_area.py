#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_url = './hw1/hw1_input/00.png'
img = Image.open(img_url)
plt.imshow(img, cmap="gray")

#%%
def scale(img_data, width, height):
    new_img_data = np.zeros((height, width), dtype=np.uint8)

    img_width = img_data.shape[1]
    img_height = img_data.shape[0]

    x_rate = img_width/width
    y_rate = img_height/height

    for i in range(height):
        begin_y = i * y_rate
        end_y = (i+1) * y_rate

        begin_sub_y = int(begin_y)
        end_sub_y = int(end_y)

        begin_rate_y = 1 - (begin_y - begin_sub_y)
        end_rate_y = end_y - end_sub_y

        end_sub_y = min(end_sub_y, img_height-1)

        for j in range(width):
            begin_x = j * x_rate
            end_x = (j+1) * x_rate


            begin_sub_x = int(begin_x)
            end_sub_x = int(end_x)

            begin_rate_x = 1 - (begin_x - begin_sub_x)
            end_rate_x = end_x - end_sub_x

            end_sub_x = min(end_sub_x, img_width-1)

            area = img_data[begin_sub_y:end_sub_y+1, begin_sub_x:end_sub_x+1]
            area_weight = np.ones(area.shape)
            
            for x in range(area.shape[0]):
                area_weight[x, 0] *= begin_rate_y
                area_weight[x, area.shape[1] - 1] *= end_rate_y

            for y in range(area.shape[1]):
                area_weight[0, y] *= begin_rate_x
                area_weight[area.shape[0] - 1, y] *= end_rate_x

            new_img_data[i][j] = np.average(area, weights=area_weight)

    return new_img_data

# %%
scale_list = np.array([
    [400, 400],
    [192, 128],
    [96, 64],
    [48, 32],
    [24, 16],
    [12, 8],
    [300, 200],
    [450, 300],
    [500, 200]
])

for scale_level in scale_list:
    # plt.figure(figsize=(9,6))
    plt.imshow(scale(np.array(img), scale_level[0], scale_level[1]), cmap='gray')
    plt.show()
    plt.imshow(img.resize(scale_level), cmap='gray')
    plt.axis("off")
    plt.show()

# %%
