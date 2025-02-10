import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
import io

class ascii_by_pca:
    def __init__(self):
        self.l1_norm = lambda array1, array2: np.sum(np.abs(array1 - array2))
        self.l2_norm = lambda array1, array2: np.sqrt(np.sum((array1 - array2) ** 2))
    
    def data_prep(self, path, font_path, new_width = 80, pca_n = 5):
        self.img_path = path
        self.img = cv.imread(self.img_path,cv.IMREAD_GRAYSCALE)
        print(f"Real Shape of Img: {self.img.shape}")
        self.height, self.width = self.img.shape

        # n_width = 8 * new_width
        # ratio = self.width / self.height
        # n_height = int(n_width / ratio)
        # self.img = cv.resize(self.img, (n_width, n_height), interpolation=cv.INTER_AREA)
        # self.height, self.width = self.img.shape

        if self.width < new_width:
            n_width = 8 * new_width
            ratio = self.width / self.height
            n_height = int(n_width / ratio)
            self.img = cv.resize(self.img, (n_width, n_height), interpolation=cv.INTER_AREA)
            self.height, self.width = self.img.shape

        if self.height < new_width:
            n_height = 8 * new_width
            ratio = self.width / self.height
            n_width = int(ratio * n_height)
            self.img = cv.resize(self.img, (n_width, n_height), interpolation=cv.INTER_AREA)
            self.height, self.width = self.img.shape

        self.tile = self.width//new_width
        print(f"Scaled Ratio of the IMG: {self.tile}")
        self.C = self.width // self.tile
        self.R = self.height // self.tile
        print(f"Dimension of New IMG: {self.R} x {self.C}")
        print(f"Patch size Dimension: {self.tile} X {self.tile}")
        # if self.img is not None:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(self.img, cmap='gray')
        #     plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=1)

        #     # Set grid intervals (spacing for grid lines)
        #     plt.xticks(range(0, self.img.shape[1], self.tile))  # Vertical lines
        #     plt.yticks(range(0, self.img.shape[0], self.tile))  # Horizontal lines
        #     plt.tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
        #     plt.title("Real Image")
        #     plt.show()
        
        self.pca_components = pca_n
        # self.pca = PCA(pca_n)
        self.font_path = font_path
        self.style = font_path.rsplit("/",1)[-1]
        self.ascii_dataset = self.ascii_prepare(self.tile)
        
        self.output_name = path.rsplit(".",1)[0]

        print(f"PCA_n: {pca_n}, font_path: {font_path}, style: {self.style}, output_name: {self.output_name}")
    
    def pca_get(self, img, n = 5):
        self.pca = PCA(n_components=min(n, min(img.shape)))
        pca_result = self.pca.fit_transform(img)
        # return pca_result.flatten()
        return pca_result

    
    # def pca_inv(self, x, n = 5):
    #     return self.pca.inverse_transform(x)
    
    def char2img (self, text, N=10):
        # Step 1: Create a blank image
        image_size = (N, N)  # 9x9 image (N x N)
        img = Image.new('L', image_size, color=0)  # 'L' mode for grayscale, background color white (255)

        # Step 2: Load Courier font
        try:
            # print(f"self.style = {self.style}")
            style = self.style
            # print(f" style = {style}")
            font = ImageFont.truetype(style, size=N*.9)  # Adjust font size to fit the image (font-size = N*0.90)
        except IOError:
            print(f"{self.style} font not found! Using default font.")
            font = ImageFont.load_default()

        # Step 3: Draw the letter "a" on the image
        draw = ImageDraw.Draw(img)
        # text = "Ͱ"
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Get the bounding box of the text
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center the text in the 9x9 image
        text_x = (image_size[0] - text_width) // 2
        text_y = (image_size[1] - text_height) // 2

        # Move the text a little up
        offset = -(N*0.15)  # Negative value moves the text up (offset = -N*0.10)
        text_y += offset

        draw.text((text_x, text_y), text, fill=255, font=font)  # Fill color black (0)

        # Step 4: Convert to NumPy array and display
        img_array = np.array(img)

        return img_array
    
    def ascii_prepare(self, tile):
        ascii_chars_1 = ''.join(chr(i) for i in range(32,126))
        ascii_chars_2 = ''.join(chr(i) for i in range(161,323))
        ascii_chars_full = ascii_chars_1 + ascii_chars_2
        print(ascii_chars_full)

        pca_dataset = []
        for i in ascii_chars_full:
            pca_result = self.pca_get(self.char2img(i,tile), self.pca_components)
            pca_dataset.append([i, pca_result])

        return pca_dataset
    
    def patch2ascii(self, img, n=5):
        ascii_dataset = self.ascii_dataset

        patch_pca = self.pca_get(img, n)

        max_len = max(patch_pca.shape[0], ascii_dataset[0][1].shape[0])

        # if patch_pca.shape[0] != ascii_dataset[0][1].shape[0]:
        #     # Pad both arrays with zeros to match the maximum length
        #     patch_pca = np.pad(patch_pca, (0, max_len - patch_pca.shape[0]), mode='constant')
        #     # padded_ascii = np.pad(ascii_dataset[0][1], (0, max_len - ascii_dataset[0][1].shape[0]), mode='constant')

        limit = len(ascii_dataset)

        min_l2 = float('inf')
        min_l2_position = 0

        for i in range(limit):
            ascii_patch = ascii_dataset[i][1]

            # Ensure both have the same shape before subtraction
            if ascii_patch.shape != patch_pca.shape:
                # print(f"Shape mismatch: patch_pca {patch_pca.shape}, ascii_patch {ascii_patch.shape}")
                continue  # Skip incompatible shapes

            l2 = self.l2_norm(patch_pca, ascii_patch)
            if l2 < min_l2:
                min_l2 = l2
                min_l2_position = i

        return ascii_dataset[min_l2_position][0]
    
    def img2ascii_file(self):
        file_out_path = f'{self.output_name}_pca_{self.pca_components}_{self.style}.txt'
        print(f"name= {file_out_path}")
        self.output_file = file_out_path
        with io.open(file_out_path, 'w', encoding='utf-8') as file:
            # file.write('Hello, world!\n')
            for i in range(0, self.height, self.tile):
                for j in range(0, self.width, self.tile):
                    char = self.patch2ascii(self.img[i:(i+self.tile), j:(j+self.tile)], self.pca_components)  # Get the tile section
                    file.write(char)
                file.write("\n")

            print(f"{file_out_path} created successfully!!")

    # Load the text file
    def load_text(self, file_path=None):
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.rstrip("\n") for line in file]
        
    # Convert text to image
    def text_to_image(self, text_lines=None, image_path=None, font_path=None, font_size=0, image_size=None):
        if text_lines == None:
            text_lines = self.load_text(self.output_file)

        if image_path == None:
            path_name = self.output_file.rsplit(".",1)[0]
            font_name = self.style
            image_path = f'{path_name}_text_{font_name}.png'
        
        if font_path == None:
            font_path = self.font_path

        if font_size == 0:
            font_size = self.tile

        if image_size == None:
            image_size = (self.width, self.height)
        
        # print(f"text = {text_lines}, font = {font_size},font_path = {font_path}, size = {image_size}")
        # Image dimensions
        width, height = image_size
        cell_width = width // self.C
        cell_height = height // self.R

        # Create an empty white image
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)

        # Set the font to Courier
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype("cour.ttf", font_size)  # Use Courier by default

        # Draw characters on the image
        for i, line in enumerate(text_lines):
            for j, char in enumerate(line):
                x = j * cell_width
                y = i * cell_height
                draw.text((x, y), char, fill="white", font=font)

        # Save the image
        img.save(image_path)
        print(f"Image saved at {image_path}")

courier = "path/to/cour.ttf"
# arial = "C:/Windows/Fonts/arial.ttf"
arial = "path/to/arial.ttf"
# times_new_roman = "C:/Windows/Fonts/times.ttf"
times_new_roman = "path/to/times.ttf"
# tahoma = "C:/Windows/Fonts/tahoma.ttf"
tahoma = "path/to/tahoma.ttf"
# calibri = "C:/Windows/Fonts/calibri.ttf"
calibri = "path/to/calibri.ttf"
# verdana = "C:/Windows/Fonts/verdana.ttf"
verdana = "path/to/verdana.ttf"

# from asg1_2 import ascii_by_pca
# obj1 = ascii_by_pca()
# img_path = input("Enter Image Path: ")
# path = "./monkeyking.jpg"
# obj1.data_prep(path, courier, 80, 5)
# obj1.img2ascii_file()
# obj1.text_to_image(font_path = tahoma)