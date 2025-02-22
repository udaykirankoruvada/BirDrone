import cv2
import subprocess
import os

# If FFmpeg is not in PATH, set the full path manually
FFMPEG_PATH = "C://Program Files (x86)//ffmpeg//bin//ffmpeg.exe"  # Change this to the full path if needed (e.g., r"C://ffmpeg//bin//ffmpeg.exe")


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()  # Ensure resources are released
    return frames


def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to save.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 format
    frame_height, frame_width = output_video_frames[0].shape[:2]

    temp_output = output_video_path.replace(".mp4", "_temp.mp4")

    out = cv2.VideoWriter(temp_output, fourcc, 24.0, (frame_width, frame_height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

    # Convert video to browser-compatible format
    if convert_to_browser_compatible(temp_output, output_video_path):
        os.remove(temp_output)  # Remove temp file only if conversion was successful


def convert_to_browser_compatible(input_path, output_path):
    try:
        command = [
            FFMPEG_PATH, "-y", "-i", input_path,
            "-vcodec", "libx264", "-acodec", "aac", "-strict", "experimental",
            output_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Debugging Output
        print("FFmpeg Output:", result.stdout)
        print("FFmpeg Error:", result.stderr)

        return True  # Conversion successful
    except FileNotFoundError:
        print("Error: FFmpeg not found! Ensure FFmpeg is installed and added to PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print("FFmpeg Execution Error:", e.stderr)
        return False
