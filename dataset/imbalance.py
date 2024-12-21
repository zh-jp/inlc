import numpy as np


def get_imb_num_per_cls(data, num_classes: int, r_imb: float) -> list:
    """
    imbalance type: 'exp'
    $n_k=n_kr^k$, r denotes the imbalance ratio, and k is the class index.
    """
    max_num = len(data) / num_classes
    num_per_cls = []
    for cls_idx in range(num_classes):
        num = max_num * (r_imb ** (cls_idx / (num_classes - 1.0)))
        num_per_cls.append(int(num))
    return num_per_cls


def gen_imb_data(data: np.ndarray, targets, num_per_cls):
    new_data, new_targets = [], []
    targets = np.array(targets, dtype=np.int32)
    classes = np.unique(targets)
    for cls, img_num in zip(classes, num_per_cls):
        idx = np.where(targets == cls)[0]
        np.random.shuffle(idx)
        selected_idx = idx[:img_num]
        new_data.append(data[selected_idx])
        new_targets.extend([cls] * img_num)

    new_data = np.vstack(new_data)
    return new_data, new_targets


def get_imb_data(data, targets, num_classes, r_imb):
    num_per_cls = get_imb_num_per_cls(data, num_classes, r_imb)
    return gen_imb_data(data, targets, num_per_cls)


def get_cls_num(targets, num_classes) -> np.ndarray:
    targets = np.array(targets, dtype=int)
    cls_num = np.zeros(num_classes, dtype=int)
    for i in range(num_classes):
        cls_num[i] = np.sum(targets == i)
    return cls_num
