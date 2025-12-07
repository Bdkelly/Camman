import cv2
import os

# Set up the directory and image file pattern
image_folder = '/Users/Ben/Documents/dever/python/ptorch/inferred_images/mlsvideo/imgs' # The directory where your frames are stored [cite: 227]
video_name = 'output_video.mp4'

# Get sorted list of images
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()

# Read the first image to get dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For MP4 files
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height)) # 30 FPS

# Loop through images and write to video
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release everything
cv2.destroyAllWindows()
video.release()