import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, filters, color, morphology
from skimage.morphology.selem import square
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from collections import Counter
import os


def test(mean, SD, train_features, train_labels, testfile_dir, show):
    Features = []
    img = io.imread(testfile_dir)
    th = filters.threshold_otsu(img)
    img_binary = (img < th).astype(np.double)
    close = morphology.closing(img_binary, square(1))
    final = close
    img_label = label(final, background=0)
    regions = regionprops(img_label)
    if show:
        io.imshow(final)
    ax = plt.gca()
    count = 0
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if maxr - minr > 10 and maxc - minc > 10:
            count += 1
            roi = final[minr:maxr, minc:maxc]
            m = moments(roi)
            cc = m[0, 1] / m[0, 0]
            cr = m[1, 0] / m[0, 0]
            mu = moments_central(roi, center=(cr, cc))
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            Features.append(hu)
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   fill=False, edgecolor='red', linewidth=1))
    if show:
        ax.set_title('Bounding Boxes')
        io.show()
    for arr in Features:
        for i in range(len(arr)):
            arr[i] = (arr[i] - mean[i]) / SD[i]
    D = cdist(Features, train_features)
    if show:
        io.imshow(D, aspect='auto')
        plt.title('Distance Matrix')
        io.show()
    D_index = np.argsort(D, axis=1)
    Ytrue = list()

    for row in D_index:
        neighbors = []
        for i in range(9):
            neighbors.append(train_labels[row[i]])
        Ytrue.append(Counter(neighbors).most_common(1)[0][0])
    # print(Ytrue)
    pkl_file = open('test_gt_py3.pkl', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict[b'classes']
    locations = mydict[b'locations']
    count = 0
    correct = 0
    characters = Counter()
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if maxr - minr > 10 and maxc - minc > 10:
            for index, location in enumerate(locations):
                if minc <= location[0] <= maxc and minr <= location[1] <= maxr:
                    if Ytrue[count].lower() == classes[index].lower():
                        if show:
                            print("Identified " + str(count) + " as " + str(classes[index]))
                        characters[Ytrue[count]] += 1
                        correct += 1
                    else:
                        if show:
                            print("Mistook " + str(classes[index]) + " for " + str(Ytrue[count]))
                    break
            count += 1
    print(characters)
    return correct / len(classes)


if __name__ == '__main__':
    test("./test.bmp")
