import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
import subprocess
import imageio_ffmpeg as ffmpeg
import os

class mp4_ascii:
    def __init__(self, new_width = 80):
        self.grey70 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^‘’. "
        self.grey10 = "@%#*+=-:. "
        self.grey70 = self.grey70[::-1]
        self.output_folder = None
        self.new_width = new_width

    def split_audio(self, input_video):
        name = input_video.rsplit(".",1)[0]
        self.output_folder = input_video.rsplit(".",1)[0]
        output_audio = f"{name}.mp3"
        self.output_audio = output_audio

        # Delete the file if it already exists
        if os.path.exists(output_audio):
            os.remove(output_audio)
            print(f"Deleted existing file: {output_audio}")

        cmd = f'ffmpeg -i "{input_video}" -q:a 0 -map a "{output_audio}"'
        ffmpeg.get_ffmpeg_exe()
        os.system(cmd)

        print("Audio extracted successfully!")
    
    def vid_frames(self, video_path):
        self.output_folder = video_path.rsplit(".",1)[0]

        # Load the video
        cap = cv.VideoCapture(video_path)
        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Cannot open the video.")
            exit()
        # Get video properties
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv.CAP_PROP_FPS)
        print(f"video_fps: {video_fps}")
        self.fps = video_fps
        
        frame_folder = f"{self.output_folder}/video_frames"
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        frame_count = 0
        # Loop through each frame
        while True:
            ret, frame = cap.read()
            # Break the loop if no frame is returned (end of video)
            if not ret:
                break
            frame_filename = os.path.join(frame_folder, f"frame_{frame_count:06d}.png")
            cv.imwrite(frame_filename, frame)
            # print(f"Saved: {frame_filename}")
            frame_count += 1

        # Release the video capture object
        cap.release()
        print(f"Extraction complete. Total frames: {frame_count}")

    def initialize(self):
        img_path = f"{self.output_folder}/video_frames/frame_000013.png"
        img = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
        print(f"Real Shape of Img: {img.shape}")
        self.height, self.width = img.shape
        self.tile = self.width // self.new_width
        print(f"Scaled Ratio of the IMG: {self.tile}")
        C = self.width // self.tile
        R = self.height // self.tile
        print(f"Dimension of New IMG: {R} x {C}")
        print(f"Patch size Dimension: {self.tile} X {self.tile}")
        if img is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(img, cmap='gray')
            plt.grid(color='white', linestyle='-', linewidth=0.5, alpha=1)

            # Set grid intervals (spacing for grid lines)
            plt.xticks(range(0, img.shape[1], self.tile))  # Vertical lines
            plt.yticks(range(0, img.shape[0], self.tile))  # Horizontal lines
            plt.tick_params(axis='both', which='both', labelleft=False, labelbottom=False)
            plt.title("Real Image")
            plt.show()
        self.cell_width = self.width // C  # Width of each cell (160 columns)
        self.cell_height = self.height // R  # Height of each cell (139 rows)

    def img2ascii(self, img_path, txt_path):
        # img_path = "./luffy_gear_5.png"
        img = cv.imread(img_path,cv.IMREAD_GRAYSCALE)

        grey10 = self.grey10
        x = len(grey10)
        with open(txt_path, 'w', encoding="utf-8") as file:
            # file.write('Hello, world!\n')
            for i in range(0, self.height, self.tile):
                for j in range(0, self.width, self.tile):
                    tile_section = img[i:(i+self.tile), j:(j+self.tile)]  # Get the tile section
                    # if tile_section.size > 0:
                    num = int(np.mean(tile_section) / (255 / x))
                    # print(num)
                    file.write(grey10[num-1])
                file.write("\n")
    
    def img2ascii70(self, img_path, txt_path):
        # img_path = "./luffy_gear_5.png"
        img = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
        img_color = cv.imread(img_path)  # Read the image in color (BGR format)

        grey70 = self.grey70
        x = len(grey70)
        with open(f"{txt_path}.txt", 'w', encoding="utf-8") as file:
            # file.write('Hello, world!\n')
            for i in range(0, self.height, self.tile):
                for j in range(0, self.width, self.tile):
                    tile_section = img[i:(i+self.tile), j:(j+self.tile)]  # Get the tile section
                    # if tile_section.size > 0:
                    num = int(np.mean(tile_section) / (255 / x))
                    # print(num)
                    file.write(grey70[num-1])
                file.write("\n")
        
        with open(f"{txt_path}_RGB.txt", 'w', encoding="utf-8") as file:
            for i in range(0, self.height, self.tile):
                for j in range(0, self.width, self.tile):
                    tile_section = img_color[i:(i+self.tile), j:(j+self.tile)]  # Extract the patch
                    if tile_section.size > 0:
                        avg_color = np.mean(tile_section, axis=(0, 1))  # Calculate average color (BGR)
                        avg_color_rgb = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))  # Convert to (R, G, B)
                        file.write(f"{avg_color_rgb} ")  # Write (R, G, B) values separated by space
                file.write("\n")  # New line for each row of patches

    # Load the text file
    def load_text(self, file_path):
        with open(f"{file_path}.txt", "r", encoding="utf-8") as file:
            return [line.rstrip("\n") for line in file]
        
    # Function to load RGB color values from the text file
    def load_rgb_text(self, file_path):
        with open(f"{file_path}_RGB.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()

        rgb_values = []
        for line in lines:
            row_colors = []
            colors = line.strip().split(") (")  # Splitting each RGB tuple
            for color in colors:
                color = color.replace("(", "").replace(")", "").replace(",", "")  # Cleaning the values
                rgb = tuple(map(int, color.split()))  # Converting to integers
                row_colors.append(rgb)
            rgb_values.append(row_colors)
        return rgb_values

    # Convert text to image
    def text_to_image(self, text_lines, rgb_values, image_path, font_path=None, font_size=12):
        # global cell_height, cell_width
        # Create an empty white image
        img = Image.new("RGB", (self.width, self.height), "black")
        draw = ImageDraw.Draw(img)

        # Set the font to Courier
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype("cour.ttf", font_size)  # Use Courier by default

        # Draw characters on the image
        for i, line in enumerate(text_lines):
            for j, char in enumerate(line):
                x = j * self.cell_width
                y = i * self.cell_height
                if i < len(rgb_values) and j < len(rgb_values[i]):  # Ensure index is within bounds
                    color = rgb_values[i][j]  # Get corresponding RGB color
                    draw.text((x, y), char, fill=color, font=font)  # Draw character with its color

        # Save the image
        img.save(image_path)
        # print(f"Image saved at {image_path}")

    def frame2ascii(self):
        folder_path = f"{self.output_folder}/video_frames"
        frame_count = len([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        arial = "path/to/arial.ttf"
        font_size = self.tile
        print("Total frames:", frame_count)

        text_folder = f"{self.output_folder}/text_frame"
        if not os.path.exists(text_folder):
            os.makedirs(text_folder)
        ascii_frame = f"{self.output_folder}/ascii_frame"
        if not os.path.exists(ascii_frame):
            os.makedirs(ascii_frame)

        frame_count = int(self.fps * 20)
        
        for i in range(frame_count):
            img_path = f"{folder_path}/frame_{i:06d}.png"
            text_path = f"{text_folder}/frame_{i:06d}"
            self.img2ascii70(img_path, text_path)

            output_image_path = f"{ascii_frame}/frame_{i:06d}.png"
            # Load the text and convert it to an image
            text_lines = self.load_text(text_path)
            rgb_data = self.load_rgb_text(text_path)
            self.text_to_image(text_lines, rgb_data, output_image_path, arial, font_size)

        
    def compressVid(self):
        # Set parameters
        fps = self.fps  # Change FPS as needed
        frame_folder = f"{self.output_folder}/ascii_frame"  # Folder containing the images
        # Path to save the output video
        output_path = f"{self.output_folder}/ascii_video"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_video_path = output_path

        output_video = f"{output_path}/output.mp4"
        if os.path.exists(output_video):
            os.remove(output_video)
        # self.output_video = output_video
        
        crf = 35  # Lower means higher quality; higher means smaller file size

        # Paths
        compressed_video = f"{output_path}/compressed_output.mp4"
        if os.path.exists(compressed_video):
            os.remove(compressed_video)
        self.output_video = compressed_video

        # Get frame list
        frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Read first frame to get dimensions
        frame = cv.imread(frames[0])
        height, width, _ = frame.shape

        # **Create Video from Frames using FFmpeg**
        ffmpeg_cmd = [
            ffmpeg.get_ffmpeg_exe(),
            "-framerate", str(fps),  # Frame rate
            "-i", os.path.join(frame_folder, "frame_%06d.png"),  # Input frames with padding (frame_000001, frame_000002, ...)
            "-c:v", "libx264",  # Video codec
            "-pix_fmt", "yuv420p",  # Ensures compatibility
            "-y", output_video  # Output file
        ]

        subprocess.run(ffmpeg_cmd)
        print("Video created successfully!")

        # **Compress Video**
        compression_cmd = [
            ffmpeg.get_ffmpeg_exe(),
            "-i", output_video,  # Input video
            "-vcodec", "libx264",
            "-crf", str(crf),  # CRF (higher = more compression)
            "-preset", "slow",  # Slower = better compression
            "-y", compressed_video  # Output file
        ]

        subprocess.run(compression_cmd)
        print("Video compressed successfully!")

        # Check if the final video with audio exists
        if os.path.exists(compressed_video):
            print(f"{compressed_video} exists. Deleting {output_video}...")
            if os.path.exists(output_video):
                os.remove(output_video)
                print(f"Deleted: {output_video}")
            else:
                print(f"{output_video} does not exist, nothing to delete.")
        else:
            print(f"{compressed_video} does not exist, keeping {output_video}.")

    def vidWithAud(self):
        video_file = self.output_video
        audio_file = self.output_audio
        output_video_with_audio = f"{self.output_video_path}/final_video.mp4"

        # FFmpeg command to merge video and audio
        cmd = (
            ffmpeg.get_ffmpeg_exe(),
            "-y",  # Overwrite existing file
            "-i", video_file,  # Input video
            "-i", audio_file,  # Input audio
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",  # Encode audio to AAC for compatibility
            "-strict", "experimental",  # Allow experimental features
            "-shortest",  # Trim to the shortest stream length (if audio is longer)
            output_video_with_audio
        )

        # Run FFmpeg command
        os.system(" ".join(cmd))

        print(f"Video with audio saved as {output_video_with_audio}")

        # Check if the final video with audio exists
        if os.path.exists(output_video_with_audio):
            print(f"{output_video_with_audio} exists. Deleting {video_file}...")
            if os.path.exists(video_file):
                os.remove(video_file)
                print(f"Deleted: {video_file}")
            else:
                print(f"{video_file} does not exist, nothing to delete.")
        else:
            print(f"{output_video_with_audio} does not exist, keeping {video_file}.")

    def fullwork(self, video_path):
        self.split_audio(video_path)
        self.vid_frames(video_path)
        self.initialize()
        self.frame2ascii()
        self.compressVid()
        self.vidWithAud()
        