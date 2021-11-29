import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology, color, filters
from skimage.morphology.selem import square, diamond, star, disk, octagon, ellipse, rectangle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os


def train(imagedir, show):
    plt.interactive(False)
    labels = []
    Features = []
    for filename in sorted(os.listdir(imagedir)):
        img = io.imread(imagedir + "/" + filename)
        th = filters.threshold_otsu(img)
        img_binary = (img < th).astype(np.double)
        close = morphology.closing(img_binary, square(1))
        final = close
        img_label = label(final, background=0)
        regions = regionprops(img_label)
        if show:
            io.imshow(final)
        ax = plt.gca()
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            if maxr - minr > 10 and maxc - minc > 10:
                roi = final[minr:maxr, minc:maxc]
                m = moments(roi)
                cc = m[0, 1] / m[0, 0]
                cr = m[1, 0] / m[0, 0]
                mu = moments_central(roi, center=(cr, cc))
                nu = moments_normalized(mu)
                hu = moments_hu(nu)
                Features.append(hu)
                labels.append(filename.split(".")[0])
                if show:
                    ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr,
                                           fill=False, edgecolor='red', linewidth=1))
                    ax.set_title('Bounding Boxes')
        if show:
            io.show()
    standard_deviation = []
    mean = []
    for i in range(len(Features[0])):
        standard_deviation.append(np.std([row[i] for row in Features]))
        mean.append(np.mean([row[i] for row in Features]))
    for arr in Features:
        for i in range(len(arr)):
            arr[i] = (arr[i] - mean[i]) / standard_deviation[i]
    return mean, standard_deviation, Features, labels


if __name__ == '__main__':
    train("./images", True)
