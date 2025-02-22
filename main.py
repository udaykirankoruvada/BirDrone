import os
import random
from utils.video_utils import read_video, save_video
from models.objectDetector.detection import Detector


def generate_unique_filename():
    # Generate a random number and create a filename
    random_number = random.randint(1000, 9999)
    output_video_path = f"output_video_{random_number}.mp4"

    # Check if the file already exists, and if so, generate a new name
    while os.path.exists(output_video_path):
        random_number = random.randint(1000, 9999)
        output_video_path = f"output_video_{random_number}.mp4"

    return output_video_path


def main():
    # frames = read_video("F://Udaykiran//FinalYearProject//BirDrone//BirdroneApp//public//videos//inputfile.mp4")
    frames = read_video("C://Users//Udaykiran//Downloads//Untitled video - Made with Clipchamp.mp4")
    # frames = read_video("C://Users//Udaykiran//Downloads//Input_video2.mp4")
    detector = Detector()

    # Generate a unique output video path
    output_video_path = generate_unique_filename()

    # Process the video
    detector.process_video(frames, output_video_path)
    print(f"Output video saved at: {output_video_path}")

if __name__ == '__main__':
    main()
