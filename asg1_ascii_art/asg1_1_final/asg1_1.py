import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def img2ascii(patch, img_path):
    img_path = "./" + img_path
    img = cv.imread(img_path,cv.IMREAD_GRAYSCALE)

    print(f"Real Shape of Img: {img.shape}")
    height, width = img.shape

    if width < patch:
        n_width = patch
        ratio = width / height
        n_height = int(ratio * patch)
        img = cv.resize(img, (n_width, n_height), interpolation=cv.INTER_AREA)
        height, width = img.shape

    if height < patch:
        n_height = patch
        ratio = width / height
        n_width = int(ratio * patch)
        img = cv.resize(img, (n_width, n_height), interpolation=cv.INTER_AREA)
        height, width = img.shape

    tile = round(width / patch) 
    print(f"Scaled Ratio of the IMG: {tile}")

    C = width // tile
    R = height // tile
    print(f"Dimension of New IMG: {R} x {C}")

    print(f"Patch size Dimension: {tile} X {tile}")

    if img is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=1)

        # Set grid intervals (spacing for grid lines)
        plt.xticks(range(0, img.shape[1], tile))  # Vertical lines
        plt.yticks(range(0, img.shape[0], tile))  # Horizontal lines
        plt.tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
        plt.title("Real Image")
        plt.show()

    grey70 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^‘’. "
    grey10 = "@%#*+=-:. "

    output_file_name = img_path.rsplit(".",1)[0]

    x = len(grey70)
    with open(f'{output_file_name}_{x}_Level_{patch}.txt', 'w') as file:
        # file.write('Hello, world!\n')
        for i in range(0, height, tile):
            for j in range(0, width, tile):
                tile_section = img[i:(i+tile), j:(j+tile)]  # Get the tile section
                # if tile_section.size > 0:
                num = int(np.mean(tile_section) / (255 / x))
                file.write(grey70[num-1])
            file.write("\n")

    x = len(grey10)
    with open(f'{output_file_name}_{x}_Level_{patch}.txt', 'w') as file:
        # file.write('Hello, world!\n')
        for i in range(0, height, tile):
            for j in range(0, width, tile):
                tile_section = img[i:(i+tile), j:(j+tile)]  # Get the tile section
                # if tile_section.size > 0:
                num = int(np.mean(tile_section) / (255 / x))
                file.write(grey10[num-1])
            file.write("\n")

img_path = input("Give the image file name with extension: ")
img2ascii(80, img_path)
img2ascii(160, img_path)