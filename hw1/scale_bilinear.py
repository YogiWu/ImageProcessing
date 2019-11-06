#%%
### hw 2.2 Scaling

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
img_url = './hw1/hw1_input/00.png'
img = mpimg.imread(img_url)

print(img.shape, img[0,0])

### Origin Image
plt.imshow(img, cmap='gray')
plt.show()

#%%
from PIL import Image

img2 = Image.open(img_url)
img_data_ = np.array(img2)

print(img_data_.shape, img_data_[0,0])
plt.imshow(img_data_, cmap='gray')
plt.show()

#%%
def bi_linear(row1, row2, coordinate):
    x = coordinate[0]
    y = coordinate[1]

    y1 = row1[0]*y + row2[0]*(1-y)
    y2 = row1[1]*y + row2[1]*(1-y)

    return round(y1*x+y2*(1-x))

#%%
def scale(img_data, width, height):
    # new_img = Image.new(img_data.mode, img_data.size, 0)
    new_img_data = np.zeros((height, width), dtype=np.int)

    img_width = img_data.shape[1]
    img_height = img_data.shape[0]

    x_rate = img_width/width
    y_rate = img_height/height

    for i in range(height):
        coordinate_y = (i+0.5)*y_rate

        round_y = int(round(coordinate_y))
        begin_y = round_y - 1
        end_y = begin_y + 1
        coordinate_y = coordinate_y - begin_y - 0.5

        begin_y = begin_y if (begin_y > 0) else 0
        end_y = end_y if (end_y < img_height-1) else img_height-1
        
        for j in range(width):
            coordinate_x = (j+0.5)*x_rate

            round_x = int(round(coordinate_x))
            begin_x = round_x - 1
            end_x = begin_x + 1
            coordinate_x = coordinate_x - begin_x - 0.5

            begin_x = begin_x if (begin_x > 0) else 0
            end_x = end_x if (end_x < img_width-1) else img_width-1

            new_img_data[i,j] = bi_linear((img_data[begin_y,begin_x], img_data[begin_y,end_x]),
                                           (img_data[end_y,begin_x], img_data[end_y,end_x]),
                                           (coordinate_x, coordinate_y))

    return new_img_data
    
#%%
def scale(img_data, width, height):
    # new_img = Image.new(img_data.mode, img_data.size, 0)
    new_img_data = np.zeros((height, width), dtype=np.int)

    img_width = img_data.shape[1]
    img_height = img_data.shape[0]

    x_rate = img_width/width
    y_rate = img_height/height

    for i in range(height):
        coordinate_y = (i+0.5)*y_rate

        begin_y = int(coordinate_y)
        end_y = begin_y + 1

        coordinate_y = coordinate_y - begin_y

        begin_y = begin_y if (begin_y > 0) else 0
        end_y = end_y if (end_y < img_height-1) else img_height-1
        
        for j in range(width):
            coordinate_x = (j+0.5)*x_rate

            begin_x = int(coordinate_x)
            end_x = begin_x + 1

            coordinate_x = coordinate_x - begin_x

            begin_x = begin_x if (begin_x > 0) else 0
            end_x = end_x if (end_x < img_width-1) else img_width-1

            new_img_data[i,j] = bi_linear((img_data[begin_y,begin_x], img_data[begin_y,end_x]),
                                           (img_data[end_y,begin_x], img_data[end_y,end_x]),
                                           (coordinate_x, coordinate_y))

    return new_img_data

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

for scale_level in scale_list:
    # plt.figure(figsize=(9,6))
    # plt.figure()
    plt.imshow(scale(img_data_, scale_level[0], scale_level[1]), cmap='gray')
    plt.show()
    plt.imshow(img2.resize(scale_level), cmap='gray')
    plt.axis("off")
    # plt.savefig("./hw1/hw1_output/"+str(scale_level[0])+"_"+str(scale_level[1])+".png")
    plt.show()

# %%
for scale_level in scale_list:
    # fig = plt.figure(figsize=(scale_level[0], scale_level[1]))
    # fig.figimage(scale(img_data_, scale_level[0], scale_level[1]))
    # plt.show()
    img_data = scale(img_data_, scale_level[0], scale_level[1])
    img_data = np.uint8(img_data)
    im=Image.fromarray(img_data, "L")
    plt.imshow(im, cmap="gray")
    plt.show()
    # im.save("./hw1/hw1_output/"+str(scale_level[0])+"_"+str(scale_level[1])+".png")
    # im.show()

# %%
import IPython.display
def showarray(a, fmt='png'):
    IPython.display.display(Image.fromarray(a, 'L'))

for scale_level in scale_list:
    showarray(np.uint8(scale(img_data_, scale_level[0], scale_level[1])))
    showarray(np.array(img2.resize(scale_level)))

# %%
