import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def compute_derivations(img1, img2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])

    fx = signal.convolve2d(img1, kernel_x, boundary='symm', mode='same') + signal.convolve2d(img2, kernel_x,
                                                                                             boundary='symm',
                                                                                             mode='same')
    fy = signal.convolve2d(img1, kernel_y, boundary='symm', mode='same') + signal.convolve2d(img2, kernel_y,
                                                                                             boundary='symm',
                                                                                             mode='same')
    ft = signal.convolve2d(img1, kernel_t, boundary='symm', mode='same') + signal.convolve2d(img2, -kernel_t,
                                                                                             boundary='symm',
                                                                                             mode='same')
    return fx, fy, ft


def lucas_kanade(img1, img2, window_size):
    half_window = int(window_size / 2)
    u = np.zeros(img1.shape)
    v = np.zeros(img2.shape)
    fx, fy, ft = compute_derivations(img1, img2)

    for i in range(half_window, img1.shape[0] - half_window):
        for j in range(half_window, img1.shape[1] - half_window):
            Ix = fx[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            Iy = fy[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            It = ft[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            A = np.transpose(np.array([Ix, Iy]))
            pinv = np.linalg.pinv(np.dot(np.transpose(A), A))
            b = np.dot(pinv, np.transpose(A))
            It = -1 * It
            U = np.dot(b, It)
            u[i, j] = U[0]
            v[i, j] = U[1]

    return u, v


def plot_quiver_uv(u, v):
    u = np.flipud(u)
    v = np.flipud(v)

    fig, ax = plt.subplots()
    plt.quiver(u, v, scale_units='xy', angles='xy', scale=0.1, width=0.001, color='blue',
               minshaft=1, minlength=0)
    plt.show()
