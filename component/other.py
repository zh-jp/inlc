from collections import defaultdict
import torch


class FeatureQueue:

    def __init__(self, num_classes, feat_dim, max_size):
        self.num_classes = num_classes
        self.max_size = max_size

        self._bank = defaultdict(lambda: torch.empty(0, feat_dim).cuda())
        self.prototypes = torch.zeros(self.num_classes, feat_dim).cuda()

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        for idx in range(self.num_classes):
            # select features by label
            cls_idx = torch.where(labels == idx)[0]
            if len(cls_idx):
                max_size = self.max_size
                # push to the memory bank
                feats_selected = features[cls_idx]
                self._bank[idx] = torch.cat([self._bank[idx], feats_selected], 0)

                # fixed size
                current_size = len(self._bank[idx])
                if current_size > max_size:
                    self._bank[idx] = self._bank[idx][current_size - max_size:]

                # update prototypes
                self.prototypes[idx, :] = self._bank[idx].mean(0)


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):

    def __init__(self, model, decay, device):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)
