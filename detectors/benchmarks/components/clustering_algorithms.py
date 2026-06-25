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
        D = np.where(D > 0.5, 1, 0)

        # Closeness to disk centre grading
        grade = np.zeros_like(D)

        # Pass kernel over images to sum all pixels within a given disk
        for i in range(64):
            for j in range(64):
                pixels_to_sum = pixels_in_radius(np.array([i, j]), R=5)
                grade[i, j] = np.sum([D[c[0], c[1]] for c in pixels_to_sum])

        keypoints = detector.detect(grade.astype(np.uint8))

        source_centres_in_each_image.append(keypoints)

        # CHECK THIS WORKS THEN ALSO IMPLEMENT LoG - call versions UNEK, UNEB (UNET + Blob), and UNELOG (U-Net + LoG)
        # - STATE WHICH PAPERS THEY ARE FROM AND COMBINE WITH UNET - CHECK THEY WORK THEN FIND A FEW MORE ALGORITHMS
        # FOR DETECTION AND USE AS YOUR BENCHMARKS

    return source_centres_in_each_image





def k_means_clustering(binary_segments):

    # IMPLEMENTED FOLLOWING PSEUDOCODE IN ID8 (MY OWN IMPLEMENTATION)

    source_centres_in_each_image = []

    for segment in binary_segments:

        D = segment[0].detach().numpy()

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

# Blob Detection Thresholds - https://opencv.org/blob-detection-using-opencv/#h-filtering-blobs
# ID8 and ID25 - see references
# Indexing with array of indices - https://stackoverflow.com/questions/19821425/how-can-i-filter-numpy-array-by-list-of-
# indices
# Numpy Thresholding - https://bobbyhadz.com/blog/python-convert-numpy-array-to-0-or-1-based-on-threshold
# Setting Blob Detection Thresholds - https://stackoverflow.com/questions/32973537/what-is-the-use-of-minrepeatability-parameter-of-simpleblobdetector-in-opencv
