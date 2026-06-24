import copy
from itertools import product
import numpy as np
from sklearn.cluster import KMeans


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
