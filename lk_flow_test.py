import cv2

from lk_flow import *
from utils import read_frames

frames = read_frames('./videos/suzie_qcif.y4m')

frame1 = frames[3]
frame2 = frames[4]
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
u, v = lucas_kanade(frame1, frame2, window_size=6)
plot_quiver_uv(u, v)