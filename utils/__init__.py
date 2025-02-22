# utils/__init__.py
if __name__ == "__main__":
    from video_utils import read_video
    from video_utils import save_video

else:
    from .video_utils import read_video
    from .video_utils import save_video
