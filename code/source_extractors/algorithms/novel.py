from .components.clustering_algorithms import k_means_clustering
from .components.machine_learning_segmentation_algorithms import random_forest_segmentation

def novel_source_extraction_algorithms(training_data, validation_data, testing_data):

    segmentation_algorithms = ["random_forest"]

    localisation_algorithms = ["kmeans"]

    classification_algorithms = ["cnn"]

    for segment in segmentation_algorithms:

        match segment:

            case "random_forest":

                segmentation_predictions = random_forest_segmentation(training_data=training_data,
                                                                      testing_data=testing_data)

            case _:

                raise NameError("Segmentation algorithm {} could not be found.".format(segment))

        for local in localisation_algorithms:

            match local:

                case "kmeans":

                    source_locations = k_means_clustering(segmentation_predictions)

            print(source_locations)

            # for classifier in classification_algorithms:
            #
            #     match classifier:
            #
            #         case "cnn":
            #
            #             source_classes = c



