import torch
import torch.nn.functional as F
from torch import Tensor


class AverageMeter:

    def __init__(self, name, fmt='.4f'):
        self.count, self.sum, self.val, self.avg = 0, 0, 0, 0
        self.name = name
        self.fmt = fmt

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        res = f"{self.name} {self.avg:{self.fmt}}"
        return res


class HeadTailAcc:
    def __init__(self, num_classes: int, fmt='.3f'):
        self.correct = torch.zeros(num_classes)
        self.cls_num = torch.zeros(num_classes)
        self.fmt = fmt
        if num_classes == 10:
            self.head, self.mid, self.tail = [0, 3], [3, 7], [7, 10]
        elif num_classes == 100:
            self.head, self.mid, self.tail = [0, 36], [36, 71], [71, 100]
        else:
            raise ValueError('num_classes not in [10, 100]')
        self.num_classes = num_classes

    def update_per_cls_acc(self, logits: Tensor, targets: Tensor):
        _, pred = logits.max(1)
        pred_one_hot = F.one_hot(pred, self.num_classes)
        targets_one_hot = F.one_hot(targets, self.num_classes)
        self.cls_num = self.cls_num + targets_one_hot.sum(dim=0).float()
        self.correct = self.correct + (pred_one_hot * targets_one_hot).sum(dim=0).float()

    def get_per_cls_acc(self):
        acc_cls = self.correct / self.cls_num

        def cal_interval_acc(begin, end):
            acc = acc_cls[begin:end].mean() * 100
            return acc

        head = cal_interval_acc(self.head[0], self.head[-1])
        mid = cal_interval_acc(self.mid[0], self.mid[-1])
        tail = cal_interval_acc(self.tail[0], self.tail[-1])

        return head, mid, tail

    def __str__(self):
        head, mid, tail = self.get_per_cls_acc()
        res = (f'Head {head:{self.fmt}}\t'
               f'Mid {mid:{self.fmt}}\t'
               f'Tail {tail:{self.fmt}}')
        return res


def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> list:
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1)  # Shape: [batch_size, maxk]
    pred = pred.t()  # Shape: [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Shape: [maxk, batch_size]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
