import cv2
import shutil
import os
import argparse

argparser = argparse.ArgumentParser(description='Reconstruct a dataset with hloc.')
argparser.add_argument('-i','--input', type=str, default="outer", help='mp4 path.')
argparser.add_argument("-n",'--frame_num', type=int, default=10, help="number of frames to extract")

def extract_frames(video_path, output_dir, frame_num=10):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at the specified path: {video_path}")
    
    # Check if the output directory already exists and has content
    if os.path.exists(output_dir) and os.listdir(output_dir):
        # Clear the output directory
        shutil.rmtree(output_dir)
        
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Calculate the frame interval based on the desired number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // frame_num
    # Initialize a counter for the extracted frames
    frame_count = 0
    
    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # Break the loop if no more frames are available
        if not ret:
            break
        
        # Extract frames at the specified interval
        if frame_count % frame_interval == 0:
            # Save the frame as an image file
            frame_path = f"{output_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
        
        # Increment the frame count
        frame_count += 1
    
    # Release the video file
    video.release()
    print(f"Extracted {frame_count} frames from the video to {output_dir}")

# Example usage
video_path = f"{argparser.parse_args().input}.mp4"
output_dir = f"datasets/{argparser.parse_args().input}/mapping/"
frame_num = argparser.parse_args().frame_num  # Extract specified number of frames

extract_frames(video_path, output_dir, frame_num)