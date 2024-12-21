import argparse
import time
import torch

from util.logging import create_logger
from util.seed import set_seed
from component.inceptionResNetV2 import InceptionResNetV2
from dataset.webvision import WebvisionDataloader
from dataset.imagenet_val import ImageNetValDataloader
from dataset.imbalance import get_cls_num

from pipeline.train import train
from pipeline.evaluate import evaluate
from pipeline.prepare import optim_and_scheduler, dataloader4training, model_queue

parser = argparse.ArgumentParser()

# dataset.webvision.py needs
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset', type=str, default='webvision')

# train needs
parser.add_argument('--warm_epochs', type=int, default=5)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--tau', type=float, default=0.75)
parser.add_argument('--tau_c_max', type=float, default=0.8)
parser.add_argument('--tau_id', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=0.7)
parser.add_argument('--t_proto', type=float, default=0.05)
parser.add_argument('--w_id', type=float, default=0.5)
parser.add_argument('--w_ood', type=float, default=0.1)
parser.add_argument('--w_con', type=float, default=1)
parser.add_argument('--loss', type=str, default=None, choices=[None, 'focal'])
parser.add_argument('--metrics', type=str, default='js', choices=['js', 'loss'])
parser.add_argument('--simple_logit', action='store_true', default=False)
parser.add_argument('--s2', action='store_true', default=False)

# this file needs
parser.add_argument('--test', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max_size', type=int, default=256)
parser.add_argument('--mark', default='', type=str)

torch.set_float32_matmul_precision('high')

def main():
    logger = create_logger(args, 'webvision')
    end = time.time()
    
    if args.test:
        test(args, logger)
    else:
        main_worker(args, logger)
    logger.info(f'Total time: {time.time() - end:.2f}s')


def test(args, logger):
    dataloaders = WebvisionDataloader(batch_size=args.batch_size)
    num_classes = dataloaders.num_classes
    dataloaders2 = ImageNetValDataloader(batch_size=args.batch_size,
                                         num_classes=num_classes)
    test_loader = dataloaders.test()
    test_loader2 = dataloaders2.test()

    model = InceptionResNetV2(num_classes).cuda()
    model.load_state_dict(torch.load(f'./checkpoint/{args.name}_model_best.pth', weights_only=True))
    evaluate(model, test_loader, logger)
    evaluate(model, test_loader2, logger)


def main_worker(args, logger):

    train_loader, test_loader, num_classes = dataloader4training(args)
    model, ema_model, queue = model_queue(args, num_classes)
    optimizer, scheduler = optim_and_scheduler(args, model)

    best_acc1, best_acc5 = 0, 0

    cls_num_ls = get_cls_num(train_loader.dataset.targets, num_classes)
    args.cls_num_ls = cls_num_ls
    logger.info(f'Cls num: {cls_num_ls}')

    for epoch in range(args.epochs):
        end = time.time()
        train(args, train_loader, model, ema_model, optimizer, logger, epoch, queue)

        if epoch % 2 == 0 or epoch > 60:
            logger.info(f'Epoch[{epoch}] Evaluate on Webvision')
            acc1, acc5 = evaluate(model, test_loader, logger)
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            logger.info(f'Best acc1: {best_acc1:.3f}, Best acc5: {best_acc5:.3f}')
            if acc1 == best_acc1:
                torch.save(model.state_dict(), f'./checkpoint/{args.name}_model_best.pth')

        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch[{epoch}] end '
                    f'Best acc1: {best_acc1:.3f}\tBest acc5: {best_acc5:.3f}\t'
                    f'lr {lr:.5f}\t'
                    f'Time {time.time() - end:.2f}s')
        scheduler.step()

    torch.save(model.state_dict(), f'./checkpoint/{args.name}_model_last.pth')

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    main()
