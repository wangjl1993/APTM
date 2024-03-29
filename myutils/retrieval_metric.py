import numpy as np
from functools import partial


def retrieval_metric(ground_thruth: np.ndarray, predicted: np.ndarray, top_k: int):
    """get hit_ratio and mean_reciprocal_rank.

    Args:
        ground_thruth (np.ndarray): shape(n, 1)
        predicted (np.ndarray): shape(n, m), n samples, m recommendations.
        top_k (int): _description_

    Returns:
        _type_: _description_
    """
    top_k = min(top_k, predicted.shape[1])
    ground_thruth = ground_thruth.reshape(-1, 1)

    x, y = np.where(predicted==ground_thruth)
    y += 1

    top_k_hit = y[y<=top_k]
    hit_num = len(top_k_hit)
    hit_ratio = hit_num / len(predicted)

    mean_reciprocal_rank = (1./top_k_hit).sum() / len(predicted)

    return hit_ratio, mean_reciprocal_rank

retrieval_metric_top5 = partial(retrieval_metric, top_k=5)
retrieval_metric_top10 = partial(retrieval_metric, top_k=10)
retrieval_metric_top20 = partial(retrieval_metric, top_k=20)


# a = np.array([str(i) for i in range(10)])
# a = np.stack([a,a,a,a,a],0)
# b = np.array(['5', '1', '2', '3', '4'])

# print(retrieval_metric(b, a, 5))