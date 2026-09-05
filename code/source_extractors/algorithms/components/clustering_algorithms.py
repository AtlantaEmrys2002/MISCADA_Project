import copy
import cv2
from itertools import product
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors
import time
import warnings


def pixels_in_radius(coordinate, R):
    # SELECTS ALL INDICES AROUND A GIVEN COORDINATE THAT LIE WITHIN A DISK OF RADIUS R

    # Creates box around centre coordinate - we then look at circle with radius equal to half the len of the
    # box's width
    x_range = np.clip(np.arange(coordinate[0] - R, coordinate[0] + R), a_min=0, a_max=63)
    y_range = np.clip(np.arange(coordinate[1] - R, coordinate[1] + R), a_min=0, a_max=63)

    potential_coords = np.unique(np.array(list(product(x_range, y_range))), axis=0)

    distances_from_centre_coord = np.linalg.norm(potential_coords - coordinate, ord=2, axis=1)

    p_c = potential_coords[(distances_from_centre_coord < R)].astype(int)

    return p_c


def blob_detection(binary_segments):
    # Clustering algorithm recommended by ID25
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 30
    # Set to 78 as that is the maximum value a pixel could hold
    params.maxThreshold = 78
    params.filterByArea = True
    # 5 pixels recommended by paper ID25
    params.minArea = 5
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.thresholdStep = 10

    detector = cv2.SimpleBlobDetector_create(params)

    source_centres_in_each_image = []

    current_segment = 0

    for segment in binary_segments:

        print(current_segment)

        # PREPARE DATA

        D = segment[0]

        if not isinstance(D, np.ndarray):
            D = D.detach().numpy()

        # Threshold image (target is 0s and 1s)

        # This is important - need to "invert" image https://stackoverflow.com/questions/53064534/simple-blob-detector-
        # does-not-detect-blobs.
        D = np.where(D < 0.5, 1, 0).astype(np.uint8)

        # Closeness to disk centre grading
        grade = np.zeros_like(D, dtype=np.uint8)

        # Pass kernel over images to sum all pixels within a given disk
        for i in range(64):
            for j in range(64):
                pixels_to_sum = pixels_in_radius(np.array([i, j]), R=5).T
                grade[i, j] = np.sum(D[pixels_to_sum[0], pixels_to_sum[1]], dtype=np.uint8)

        source_centres_in_each_image.append(detector.detect(grade))

        current_segment += 1

    # N.B. we index 1 then 0, as coordinates are returned from p.pt in (y, x) format (rather than x, y)
    centres = [np.round((np.array([[p.pt[1], p.pt[0]] for p in source_centres_in_each_image[k]]))).astype(
        np.uint8) for k in range(len(source_centres_in_each_image))]

    return centres


def dbscan_clustering(binary_segments, threshold=0.2, tune=False, masks=None):

    source_centres_in_each_image = []

    if tune:

        leaf_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 200]

        total_scores = []

        times = []

        for l in leaf_sizes:

            current_index = 0

            total_score = 0

            start = time.time()

            for segment in binary_segments:

                D = segment[0] if isinstance(segment[0], np.ndarray) else segment[0].detach().numpy()

                # As we have used SoftMax, our image isn't exactly binary - this will make it so
                source_pixels = np.argwhere(D > threshold)

                if source_pixels.size != 0:

                    mask_centres = np.array([[i, j] for i in range(3, 60) for j in range(3, 60)
                                             if np.all(masks[current_index][i, j - 3: j + 4] == 1.) and
                                             np.all(masks[current_index][i - 3: i + 4, j] == 1.)])

                    labelled_source_pixels = DBSCAN(eps=5, min_samples=10, leaf_size=l).fit(
                        source_pixels).labels_

                    num_clusters_found = np.max(labelled_source_pixels)

                    cluster_centres = np.array([np.round(np.mean(np.array([k[0] for k in source_pixels[
                        np.argwhere(labelled_source_pixels == c)]]).T, axis=1)).astype(int) for c in range(1,
                                                                                                           num_clusters_found)])

                    # source_centres_in_each_image.append(np.array(cluster_centres))

                    # Find the sum of the distance between each cluster centre and its nearest neighbours
                    total_score += sum(
                        np.min(np.sqrt((cluster_centres[:, 0] - k[0]) ** 2 + (cluster_centres[:, 1] - k[1]) ** 2)) for k
                        in mask_centres)

                else:

                    source_centres_in_each_image.append(np.array([]))

                current_index += 1

            total_scores.append(total_score)

            times.append(time.time() - start)

        print("SCORES: {}".format(total_scores))
        print("TIMES: {}".format(times))
        print("Best Leaf Size for DBSCAN (in terms of error): {}".format(leaf_sizes[np.argmin(total_scores)]))
        print("Best Leaf Size for DBSCAN (in terms of time): {}".format(leaf_sizes[np.argmin(times)]))
        print("Best Time for DBSCAN: {}".format(times[np.argmin(times)]))

    # 200 is the best found during tuning
    leaf_size = leaf_sizes[np.argmin(times)] if tune else 200

    for segment in binary_segments:

        D = segment[0] if isinstance(segment[0], np.ndarray) else segment[0].detach().numpy()

        # import matplotlib.pyplot as plt
        #
        # plt.imshow(D)
        #
        # plt.show()

        # As we have used SoftMax, our image isn't exactly binary - this will make it so
        source_pixels = np.argwhere(D > threshold)

        if source_pixels.size != 0:

            labelled_source_pixels = DBSCAN(eps=5, min_samples=10, leaf_size=leaf_size).fit(source_pixels).labels_

            num_clusters_found = np.max(labelled_source_pixels)

            cluster_centres = [np.round(np.mean(np.array([k[0] for k in source_pixels[
                np.argwhere(labelled_source_pixels == c)]]).T, axis=1)).astype(int) for c in range(1,
                                                                                                   num_clusters_found)]

            source_centres_in_each_image.append(np.array(cluster_centres))


        else:

            source_centres_in_each_image.append(np.array([]))

    return source_centres_in_each_image


def k_means_clustering(binary_segments, max_num_centroids=50, threshold=0.2):
    # IMPLEMENTED FOLLOWING PSEUDOCODE IN ID8 (MY OWN IMPLEMENTATION)

    source_centres_in_each_image = []

    for segment in binary_segments:

        if isinstance(segment[0], np.ndarray):

            D = segment[0]

        else:

            D = segment[0].detach().numpy()

        # l_sth = 0.2
        l_snn = -10
        R = 5

        # As we have used SoftMax, our image isn't exactly binary - this will make it so
        V_D = np.argwhere(D > threshold)

        # Best score so far
        s_k_max = 0

        # Centre of each cluster determined by k_best
        best_centres = []

        # Determine the number of sources/clusters present in the image - do not attempt to fit more clusters than there
        # are pixels classified as source

        for k in range(1, min(max_num_centroids, int(V_D.shape[0]))):

            D_tmp = copy.deepcopy(D)

            s_k = 0

            # Elkan is more efficient, but cannot be applied when k = 1
            if k == 1:
                k_centroids = KMeans(n_clusters=k, random_state=0).fit(V_D)
            else:
                k_centroids = KMeans(n_clusters=k, random_state=0, algorithm='elkan').fit(V_D)

            # Round cluster centres (as indexing onto an image)
            cluster_centres = np.round(k_centroids.cluster_centers_)

            for c in cluster_centres:
                # REPLACE THIS WITH PIXELS IN RADIUS FUNCTION

                # find pixels of D inside R

                # Creates box around centre coordinate - we then look at circle with radius equal to half the len of the
                # box's width
                x_range = np.clip(np.arange(c[0] - R, c[0] + R), a_min=0, a_max=63)
                y_range = np.clip(np.arange(c[1] - R, c[1] + R), a_min=0, a_max=63)

                potential_coords = np.unique(np.array(list(product(x_range, y_range))), axis=0)

                distances_from_centre_coord = np.linalg.norm(potential_coords - c, ord=2, axis=1)

                p_c = potential_coords[(distances_from_centre_coord < R)].astype(int).T

                # Update s_k
                s_k += np.sum(D_tmp[p_c[0], p_c[1]])

                # Redefine scores such that if the points are included in another cluster, the score is penalised
                D_tmp[p_c[0], p_c[1]] = l_snn

            if s_k > s_k_max:
                s_k_max = s_k
                best_centres = cluster_centres

        source_centres_in_each_image.append(np.array(best_centres))

    return source_centres_in_each_image


def spectral_clustering(binary_segments, max_num_centroids=20, threshold=0.2):
    source_centres_in_each_image = []

    l_snn = -10
    R = 5

    current_index = 0

    for segment in binary_segments:

        current_index += 1

        D = segment[0]

        if not isinstance(D, np.ndarray):
            D = D.detach().numpy()

        # As we have used SoftMax, our image isn't exactly binary - this will make it so
        V_D = np.argwhere(D > threshold).astype(np.float32)

        # Best score so far
        s_k_max = 0

        # Centre of each cluster determined by k_best
        best_centres = []

        # Determine the number of sources/clusters present in the image - do not attempt to fit more clusters than there
        # are pixels classified as source

        for k in range(1, min(max_num_centroids, int(V_D.shape[0]))):

            D_tmp = copy.deepcopy(D)

            s_k = 0

            try:

                # Perform spectral clustering - chose cluster_qr, as it has no iterations or tuning parameters (and can
                # outperform the K-means algorithm). Chose 0.5 as recommended by link below (see Gamma Choice)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spectral_classifications = SpectralClustering(n_clusters=k, random_state=0, n_jobs=4, gamma=1,
                                                                  assign_labels="kmeans").fit_predict(V_D)

                # Get centre of each cluster
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cluster_centres = np.array([np.round(np.mean(V_D[np.nonzero(spectral_classifications == x)].T,
                                                                 axis=1)) for x in range(k)])

                if np.sum(np.isnan(cluster_centres)) > 0:

                    continue

                else:

                    for c in cluster_centres:
                        # REPLACE THIS WITH PIXELS IN RADIUS FUNCTION

                        # find pixels of D inside R

                        # Creates box around centre coordinate - we then look at circle with radius equal to half the len of the
                        # box's width
                        x_range = np.clip(np.arange(c[0] - R, c[0] + R), a_min=0, a_max=63)
                        y_range = np.clip(np.arange(c[1] - R, c[1] + R), a_min=0, a_max=63)

                        potential_coords = np.unique(np.array(list(product(x_range, y_range))), axis=0)

                        distances_from_centre_coord = np.linalg.norm(potential_coords - c, ord=2, axis=1)

                        p_c = potential_coords[(distances_from_centre_coord < R)].astype(int).T

                        # Update s_k
                        s_k += np.sum(D_tmp[p_c[0], p_c[1]])

                        # Redefine scores such that if the points are included in another cluster, the score is penalised
                        D_tmp[p_c[0], p_c[1]] = l_snn

                    if s_k > s_k_max:
                        s_k_max = s_k
                        best_centres = cluster_centres

            except ValueError:

                continue

        source_centres_in_each_image.append(np.array(best_centres))

    print("DONE")

    return source_centres_in_each_image

# REFERENCES

# Blob Detector Error - https://stackoverflow.com/questions/53064534/simple-blob-detector-does-not-detect-blobs
# Blob Detection OpenCV - https://pythonpathfinders.medium.com/blob-detection-in-images-using-python-computer-vision-
# 30ec881f71de
# Blob Detector Params - https://stackoverflow.com/questions/8076889/how-to-use-opencv-simpleblobdetector
# Blob Detection Thresholds - https://opencv.org/blob-detection-using-opencv/#h-filtering-blobs
# Blob Detection Tutorials - https://opencv.org/blob-detection-using-opencv/
# Blob Detection - https://www.geeksforgeeks.org/python/blob-detection-using-opencv/
# Centroids of DBSCAN - https://stackoverflow.com/questions/62215910/how-to-get-the-centroids-in-dbscan-sklearn
# Copy - https://stackoverflow.com/questions/37593013/deep-copy-of-a-np-array-of-np-array
# Extracting Cartesian Coordinates from OpenCV - https://stackoverflow.com/questions/35884409/how-to-extract-x-y-
# Filter Numpy - https://stackoverflow.com/questions/19821425/how-can-i-filter-numpy-array-by-list-of-indices
# coordinates-from-opencv-cv2-keypoint-object
# fit vs fit_predict - https://stackoverflow.com/questions/57234414/what-is-the-difference-between-fit-and-fit-predict-
# in-spectralclustering
# Gamma Choice - https://mcpanalytics.ai/articles/spectral-clustering-practical-guide-for-data-driven-decisions
# ID8 and ID25 - see references
# Indexing with array of indices - https://stackoverflow.com/questions/19821425/how-can-i-filter-numpy-array-by-list-of-
# indices
# isinstance - https://www.w3schools.com/python/ref_func_isinstance.asp
# Itertools - https://stackoverflow.com/questions/33282369/convert-itertools-array-into-numpy-array
# Mean of Rows - https://stackoverflow.com/questions/67348256/compute-the-mean-of-each-row-in-a-numpy-matrix
# Modify - https://stackoverflow.com/questions/7761393/how-to-modify-a-2d-numpy-array-at-specific-locations-without-a-
# loop
# Numpy Array Size - https://stackoverflow.com/questions/11295609/how-can-i-check-whether-a-numpy-array-is-empty-or-not
# Numpy Documentation - https://numpy.org/devdocs/reference/generated/numpy.array_equal.html
# Numpy and OpenCV Datatypes - https://stackoverflow.com/questions/7587490/converting-numpy-array-to-opencv-array
# Numpy and OpenCV Datatypes - https://stackoverflow.com/questions/54446621/why-wont-opencv-show-an-image-stored-in-a-
# numpy-array-python?rq=3
# Numpy Thresholding - https://bobbyhadz.com/blog/python-convert-numpy-array-to-0-or-1-based-on-threshold
# One-Hot Encoding - https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
# Permutations - https://stackoverflow.com/questions/1953194/permutations-of-two-lists-in-python
# Row Selection - https://stackoverflow.com/questions/58079075/numpy-select-rows-based-on-condition
# Setting Blob Detection Thresholds - https://stackoverflow.com/questions/32973537/what-is-the-use-of-minrepeatability-
# parameter-of-simpleblobdetector-in-opencv
# Spectral Error - https://stackoverflow.com/questions/38538608/sklearn-spectral-clustering-error-in-fit-predict-k-must-
# be-between-1-and-the-o
# Speed Up K-Means - https://stackoverflow.com/questions/46515481/how-to-speed-up-k-means-from-scikit-learn
# Warnings - https://stackoverflow.com/questions/3891804/raise-warning-in-python-without-interrupting-program
