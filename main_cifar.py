import argparse
import time
import numpy as np
import torch

from pipeline.train import train
from pipeline.evaluate import evaluate
from component.loss import FocalLoss
from pipeline.prepare import optim_and_scheduler, dataloader4training, model_queue

from dataset.imbalance import get_cls_num
from util.logging import create_logger
from util.seed import set_seed


parser = argparse.ArgumentParser()

# dataset.cifar.py needs
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset', type=int, choices=[10, 100], default=10)
parser.add_argument('--r_id', type=float, choices=[0, 0.1, 0.2], default=0.1)
parser.add_argument('--r_ood', type=float, choices=[0, 0.1, 0.2], default=0.1)
parser.add_argument('--r_imb', type=float, choices=[0, 0.1, 0.02, 0.01], default=0.1)
parser.add_argument('--asym', action='store_true', default=False)

# train needs
parser.add_argument('--warm_epochs', type=int, default=10) # if cifar 100 r_id=0.2, r_ood=0.2, r_imb=0.1: warm_epochs=20 is better
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--tau', type=float, default=0.75) # if cifar100, tau=0.6 is better
parser.add_argument('--tau_c_max', type=float, default=0.95)
parser.add_argument('--tau_id', type=float, default=0.4)
parser.add_argument('--eps', type=float, default=0.6)
parser.add_argument('--t_proto', type=float, default=0.05)
parser.add_argument('--w_id', type=float, default=0.5)
parser.add_argument('--w_ood', type=float, default=0.1)
parser.add_argument('--w_con', type=float, default=1)
parser.add_argument('--loss', type=str, default=None, choices=[None, 'focal'])
parser.add_argument('--metrics', type=str, default='js', choices=['js', 'loss'])
parser.add_argument('--simple_logit', action='store_true', default=False)
parser.add_argument('--s2', action='store_true', default=False)

# this file needs
parser.add_argument('--save', action='store_true', default=False)

parser.add_argument('--reweight', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--backbone', type=str, default='preresnet18', choices=['preresnet18', 'resnet32'])
parser.add_argument('--max_size', type=int, default=256)
parser.add_argument('--mark', default='', type=str)

parser.add_argument('--gamma', type=float, default=2)

torch.set_float32_matmul_precision('high')

def re_balance(args, dataloader, logger):
    targets = dataloader.dataset.targets
    num_classes = dataloader.dataset.num_classes
    cls_num_ls = get_cls_num(targets, num_classes)
    logger.info(f'Cls num: {cls_num_ls}')
    args.cls_num_ls = cls_num_ls
    if args.reweight:
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_ls)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_ls)
        per_cls_weights = torch.from_numpy(per_cls_weights).cuda()
    else:
        per_cls_weights = None

    if args.loss == 'focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=args.gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=per_cls_weights)
    return criterion

def main():
    logger = create_logger(args)
    end = time.time()

    
    main_worker(args, logger)
    logger.info(f'Total time: {time.time() - end:.2f}s')

def main_worker(args, logger):

    train_loader, test_loader, num_classes = dataloader4training(args)
    model, ema_model, queue = model_queue(args, num_classes)
    optimizer, scheduler = optim_and_scheduler(args, model)

    best_acc1, best_acc5 = 0, 0
    criterion = re_balance(args, train_loader, logger)


    for epoch in range(args.epochs):

        end = time.time()
        train(args, train_loader, model, ema_model, optimizer, logger, epoch, queue, criterion)


        acc1, acc5 = evaluate(model, test_loader, logger)
        scheduler.step()
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch[{epoch}] end '
                    f'Best acc1: {best_acc1:.3f}\tBest acc5: {best_acc5:.3f}\t'
                    f'lr {lr:.5f}\t'
                    f'Time {time.time() - end:.2f}s')

        if args.save and acc1 == best_acc1:
            logger.info(f'Saving model with best acc1: {acc1:.3f}')
            torch.save(model.state_dict(), f'./checkpoint/{args.name}_model_best.pth')

    if args.save:
        torch.save(model.state_dict(), f'./checkpoint/{args.name}_last.pth')


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    main()
