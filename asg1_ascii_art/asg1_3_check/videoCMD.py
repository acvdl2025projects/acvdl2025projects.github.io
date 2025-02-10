import os
import time
import glob

# Set the folder containing text files
folder_path = "./bleach_lr/text_frame"  # Change this to your folder path

# Get all text files in the folder
text_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

def print_file(file_path):
    """Reads and prints the content of a text file, then clears the screen."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
    with open(file_path, 'r', encoding='utf-8') as file:
        print(file.read())  # Print file content
    time.sleep(0.05)  # Pause for 2 seconds before moving to the next file

# Loop through all text files and display them one by one
for file in text_files:
    print_file(file)
