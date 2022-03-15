from scipy.spatial import distance_matrix
import numpy as np


def evaluate_points_in_image(ground_truth_points, predicted_points, valid_distance):
    all_distances = distance_matrix(ground_truth_points, predicted_points)
    sorted_indicies = np.argsort(all_distances)
    sorted_distances = np.sort(all_distances)

    # True positives
    accepted_gt = np.nonzero(sorted_distances[:, 0] <= valid_distance)[0]
    paired_pred_indicies = sorted_indicies[:, 0][accepted_gt]

    # False positives
    pred_indicies = list(range(0, len(predicted_points)))
    extra_predicted_indicies = np.setdiff1d(pred_indicies, paired_pred_indicies)

    # False negative
    gt_indicies = list(range(0, len(ground_truth_points)))
    unpredicted_indicies = np.setdiff1d(gt_indicies, accepted_gt)

    true_positive_count = len(accepted_gt)
    false_positive_count = len(extra_predicted_indicies)
    false_negative_count = len(unpredicted_indicies)
    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + false_negative_count)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1


def convert_boxes_to_points(boxes):
    boxes = np.array(boxes)
    x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    xs = ((x2s + x1s) / 2).astype(int)
    ys = ((y2s + y1s) / 2).astype(int)

    return list(zip(xs, ys))
