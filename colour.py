import numpy as np
import scipy
import random
import utils
import cv2
import bisect
from sklearn.cluster import KMeans


class ColorPalette:
    def __init__(self, colors, base_len=0):
        self.colors = colors
        self.base_len = base_len if base_len > 0 else len(colors)

    # Calculates k-means
    @staticmethod
    def from_image(img, n, max_img_size=200, n_init=8):
        img = utils.limit_size(img, max_img_size)
        clt = KMeans(n_clusters=n, n_jobs=1, n_init=n_init, algorithm="full")
        clt.fit(img.reshape(-1, 3))

        return ColorPalette(clt.cluster_centers_)

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, item):
        return self.colors[item]


# Increases brightness and saturation of the image
def increase_brightness_and_saturation(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    s[s > lim] = 255
    s[s <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# Computes the probabilities of a colour being chosen for a stroke
# A colour that is closer to the underlying colour of the stroke will have a higher probability
def compute_color_probabilities(pixels, palette, k=9):
    distances = scipy.spatial.distance.cdist(pixels, palette.colors)
    maxima = np.amax(distances, axis=1)
    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    distances = np.exp(k * len(palette) * distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    return np.cumsum(distances, axis=1, dtype=np.float32)


# Chooses the colour for the stroke
def color_select(probabilities, palette):
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]
