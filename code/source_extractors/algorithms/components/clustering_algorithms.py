import copy
import cv2
from itertools import product
import numpy as np
from sklearn.cluster import KMeans


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
    # 5 pixels recommended by paper
    params.minArea = 5
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.thresholdStep = 10

    detector = cv2.SimpleBlobDetector_create(params)

    source_centres_in_each_image = []

    for segment in binary_segments:

        # PREPARE DATA

        D = segment[0].detach().numpy().astype(np.uint8)

        # Threshold image (target is 0s and 1s)

        # This is important - need to "invert" image https://stackoverflow.com/questions/53064534/simple-blob-detector-
        # does-not-detect-blobs.
        D = np.where(D < 0.5, 1, 0)

        # Closeness to disk centre grading
        grade = np.zeros_like(D)

        # Pass kernel over images to sum all pixels within a given disk
        for i in range(64):
            for j in range(64):
                pixels_to_sum = pixels_in_radius(np.array([i, j]), R=5)
                grade[i, j] = np.sum([D[c[0], c[1]] for c in pixels_to_sum])

        grade = grade.astype(np.uint8)

        keypoints = detector.detect(grade)

        source_centres_in_each_image.append(keypoints)

    # Format centres - directly call x and y otherwise coordinates are formatted as (y, x)

    centres = []

    # N.B. we index 1 then 0, as coordinates are returned from p.pt in (y, x) format (rather than x, y)
    for k in range(len(source_centres_in_each_image)):
        centres.append(
            np.round((np.array([np.array([p.pt[1], p.pt[0]]) for p in source_centres_in_each_image[k]]))).astype(
                np.uint8))

    return centres


def k_means_clustering(binary_segments):
    # IMPLEMENTED FOLLOWING PSEUDOCODE IN ID8 (MY OWN IMPLEMENTATION)

    source_centres_in_each_image = []

    for segment in binary_segments:

        if type(segment[0]) != np.ndarray:

            D = segment[0].detach().numpy()

        else:

            D = segment[0]

        l_sth = 0.2
        l_snn = -10
        R = 5

        # As we have used SoftMax, our image isn't exactly binary - this will make it so
        V_D = np.argwhere(D > l_sth)

        # Best score so far
        s_k_max = 0

        # Centre of each cluster determined by k_best
        best_centres = []

        # Determine the number of sources/clusters present in the image
        for k in range(1, 50):

            D_tmp = copy.deepcopy(D)

            s_k = 0

            k_centroids = KMeans(n_clusters=k, random_state=0).fit(V_D)

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

                p_c = potential_coords[(distances_from_centre_coord < R)].astype(int)

                # Update s_k
                s_k += np.sum(D_tmp[p_c[:, 0], p_c[:, 1]])

                # Redefine scores such that if the points are included in another cluster, the score is penalised
                for p_c_coord in p_c:
                    D_tmp[p_c_coord[0], p_c_coord[1]] = l_snn

            if s_k > s_k_max:
                s_k_max = s_k
                best_centres = cluster_centres

        source_centres_in_each_image.append(best_centres)

    return source_centres_in_each_image


# REFERENCES

# Blob Detector Error - https://stackoverflow.com/questions/53064534/simple-blob-detector-does-not-detect-blobs
# Blob Detection OpenCV - https://pythonpathfinders.medium.com/blob-detection-in-images-using-python-computer-vision-
# 30ec881f71de
# Blob Detector Params - https://stackoverflow.com/questions/8076889/how-to-use-opencv-simpleblobdetector
# Blob Detection Thresholds - https://opencv.org/blob-detection-using-opencv/#h-filtering-blobs
# Blob Detection Tutorials - https://opencv.org/blob-detection-using-opencv/
# Blob Detection - https://www.geeksforgeeks.org/python/blob-detection-using-opencv/
# Copy - https://stackoverflow.com/questions/37593013/deep-copy-of-a-np-array-of-np-array
# Extracting Cartesian Coordinates from OpenCV - https://stackoverflow.com/questions/35884409/how-to-extract-x-y-
# Filter Numpy - https://stackoverflow.com/questions/19821425/how-can-i-filter-numpy-array-by-list-of-indices
# coordinates-from-opencv-cv2-keypoint-object
# ID8 and ID25 - see references
# Indexing with array of indices - https://stackoverflow.com/questions/19821425/how-can-i-filter-numpy-array-by-list-of-
# indices
# Itertools - https://stackoverflow.com/questions/33282369/convert-itertools-array-into-numpy-array
# Numpy Array Size - https://stackoverflow.com/questions/11295609/how-can-i-check-whether-a-numpy-array-is-empty-or-not
# Numpy Documentation - https://numpy.org/devdocs/reference/generated/numpy.array_equal.html
# Numpy and OpenCV Datatypes - https://stackoverflow.com/questions/7587490/converting-numpy-array-to-opencv-array
# Numpy and OpenCV Datatypes - https://stackoverflow.com/questions/54446621/why-wont-opencv-show-an-image-stored-in-a-
# numpy-array-python?rq=3
# Numpy Thresholding - https://bobbyhadz.com/blog/python-convert-numpy-array-to-0-or-1-based-on-threshold
# Permutations - https://stackoverflow.com/questions/1953194/permutations-of-two-lists-in-python
# Row Selection - https://stackoverflow.com/questions/58079075/numpy-select-rows-based-on-condition
# Setting Blob Detection Thresholds - https://stackoverflow.com/questions/32973537/what-is-the-use-of-minrepeatability-
# parameter-of-simpleblobdetector-in-opencv
