import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.cifar import CifarDataloader
from dataset.webvision import WebvisionDataloader
from component import EncoderClassifier
from component.other import FeatureQueue, ExponentialMovingAverage as EMAModel
from component.inceptionResNetV2 import InceptionResNetV2


def optim_and_scheduler(args, model):
    weight_decay = 1e-4
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                weight_decay=weight_decay, momentum=momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)
    return optimizer, scheduler


def dataloader4training(args):
    if args.dataset in [10, 100]:
        dataloaders = CifarDataloader(args)

    elif args.dataset == 'webvision':
        dataloaders = WebvisionDataloader(batch_size=args.batch_size)
    else:
        raise NotImplementedError(f'{args.dataset} not supported')

    train_loader = dataloaders.train(dual=True)
    test_loader = dataloaders.test()
    num_classes = dataloaders.num_classes
    return train_loader, test_loader, num_classes

def get_model(args, num_classes):
    if args.dataset in [10, 100]:
        model = EncoderClassifier(args.backbone, num_classes)
    elif args.dataset == 'webvision':
        model = InceptionResNetV2(num_classes)
    else:
        raise NotImplementedError(f'{args.dataset} not supported')
    return model

def model_queue(args, num_classes):
    model = get_model(args, num_classes).cuda()
    ema_model = EMAModel(model, 0.999, device=0)
    dim = model.dim
    queue = FeatureQueue(num_classes, dim, args.max_size)
    return model, ema_model, queue