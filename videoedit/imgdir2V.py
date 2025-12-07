import cv2
import os

def create_video_from_images(image_folder, video_name='output.mp4', fps=30):
    """
    Creates a video from a directory of images.

    Args:
        image_folder (str): Path to the folder containing images.
        video_name (str): Name of the output video file (e.g., 'my_video.mp4').
        fps (int): Frames per second for the output video.
    """
    try:
        images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
        images.sort()  # Sort the images to ensure they are in the correct order

        if not images:
            print("No images found in the specified directory.")
            return

        # Read the first image to get dimensions
        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        height, width, _ = frame.shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # This codec works for .mp4 files
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # Write each image to the video file
        print("Starting video creation...")
        for image_name in images:
            image_path = os.path.join(image_folder, image_name)
            frame = cv2.imread(image_path)
            video.write(frame)
            print(f"Adding frame: {image_name}")

        # Release the video writer object
        video.release()
        print(f"Video '{video_name}' created successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
if __name__ == '__main__':
    # Replace 'path/to/your/images' with the actual path to your image directory
    image_directory = '/Users/Ben/Documents/dever/python/ptorch/ball_tracking/mlsSmall'
    output_video_file = 'SmallVideo.mp4'
    frames_per_second = 29

    create_video_from_images(image_directory, output_video_file, frames_per_second)
