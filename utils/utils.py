import skvideo.io


def read_frames(file):
    reader = skvideo.io.FFmpegReader(file, inputdict={}, outputdict={})
    frames = []
    for frame in reader.nextFrame():
        frames.append(frame)
    return frames
