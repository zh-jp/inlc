from torch import nn, Tensor

from component.resnet import model_dict


class Classifier(nn.Module):

    def __init__(
            self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class EncoderClassifier(nn.Module):

    def __init__(self, name, num_classes=10):
        super().__init__()

        model_fun, dim = model_dict[name]
        self.encoder = model_fun()
        self.classifier = Classifier(dim, num_classes)
        self.dim = dim

    def forward(self, x, return_feat=False):
        feat = self.encoder(x)
        if return_feat:
            return feat
        logits = self.classifier(feat)
        return logits
