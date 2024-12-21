import time
import torch
import torch.nn.functional as F

from util.metrics import AverageMeter, accuracy, HeadTailAcc


@torch.no_grad()
def evaluate(model, dataloader, logger):
    model.eval()
    end = time.time()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    num_classes = dataloader.dataset.num_classes

    if num_classes in [10, 100]:
        head_tail_acc = HeadTailAcc(num_classes)
    else:
        head_tail_acc = None

    for x, labels in dataloader:
        x = x.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, labels)

        acc1, acc5 = accuracy(logits.cpu(), labels.cpu(), topk=(1, 5))
        batch_size = x.size(0)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)
        losses.update(loss.item(), batch_size)

        if head_tail_acc:
            head_tail_acc.update_per_cls_acc(logits.cpu(), labels.cpu())

    logger.info(f'Evaluate Summary '
                f'Time {time.time() - end:.2f}s\t'
                f'{losses}\t {top1}\t {top5}')
    if head_tail_acc:
        logger.info(f'{head_tail_acc}')
    return top1.avg, top5.avg
