import time
import torch.nn.functional as F
import torch


from util.metrics import AverageMeter


def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


@torch.no_grad()
def get_semantic_pseudo_label(feats_weak, queue, t_proto):
    prototypes = queue.prototypes
    sim_weak = F.cosine_similarity(feats_weak.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
    semantic_labels = F.softmax(sim_weak / t_proto, dim=1)  # shape: (B, C)

    return semantic_labels

@torch.no_grad()
def divide_ood(args, logits1, logits2):
    freq = torch.tensor(args.cls_num_ls / args.cls_num_ls.sum())
    if args.simple_logit is False:
        logit_adjust = torch.log(freq).cuda()
        logits1 = logits1 - logit_adjust
        logits2 = logits2 - logit_adjust
    pred1 = logits1.max(dim=1).indices
    pred2 = logits2.max(dim=1).indices
    ood_mask = pred1.ne(pred2)
    return ood_mask

def consistency_reg(prob1, prob2, ood_mask):
    batch_size = prob1.size(0)
    sign = torch.ones(batch_size).cuda()
    sign[ood_mask] *= -1
    loss_con = (symmetric_kl_div(prob1, prob2) * sign).mean()
    return loss_con

@torch.no_grad()
def ce_loss(logits, labels):
    loss = F.cross_entropy(logits, labels, reduction='none')
    norm_loss = (loss - loss.min()) / (loss.max() - loss.min())
    return norm_loss

def label_smoothing(targets, num_classes, eps):
    batch_size = len(targets)
    given_labels = torch.full(size=(batch_size, num_classes), fill_value=eps / (num_classes - 1))
    given_labels = given_labels.scatter(1, targets.view(-1, 1), 1 - eps).cuda(non_blocking=True)
    return given_labels

def warmup_model(model, x, labels):
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    return loss

def cal_loss_ood(ood_mask, logits, num_classes):
    num = ood_mask.sum()
    if num > 0:
        labels = torch.full((num, num_classes), 1 / num_classes).cuda()
        loss = F.cross_entropy(logits[ood_mask], labels)
    else:
        loss = 0
    return loss

def cal_loss_cls(logits, logits2, labels, criterion):
    loss = criterion(logits, labels) + criterion(logits2, labels)
    return loss

def gen_pseudo_label(feat, logits, mask, queue, t_proto, w_id):
    sem_labels = get_semantic_pseudo_label(feat[mask], queue, t_proto)
    ema_labels = F.softmax(logits[mask], dim=1)
    pse_labels = w_id * ema_labels + (1 - w_id) * sem_labels

    return pse_labels, sem_labels, ema_labels

def gen_mask(args, logits, logits_ema, prob, labels, tau_clean):
    if args.s2:
        prob_clean = 1 - js_div(prob, labels)
        clean_mask = prob_clean.ge(tau_clean)
        unclean_mask = ~clean_mask
        ood_mask_ = divide_ood(args, logits, logits_ema)
        ood_mask = ood_mask_ & unclean_mask
        id_noise_mask = ~ood_mask_ & unclean_mask
        high_conf_mask = prob.max(dim=1).values.ge(args.tau_id) & id_noise_mask
    else:
        ood_mask = divide_ood(args, logits, logits_ema)
        id_mask = ~ood_mask

        if args.metrics == 'js':
            prob_clean = 1 - js_div(prob, labels)
        else:
            prob_clean = 1 - ce_loss(logits, labels)

        clean_mask = prob_clean.ge(tau_clean) & id_mask
        id_noise_mask = prob_clean.lt(tau_clean) & id_mask
        high_conf_mask = prob.max(dim=1).values.ge(args.tau_id) & id_noise_mask

    return ood_mask, clean_mask, id_noise_mask, high_conf_mask


def train(args, dataloader, model, ema_model,
          optimizer, logger, epoch, queue, criterion=None):
    loss_meter = AverageMeter('Loss')
    end = time.time()
    model.train()
    ema_model.train()

    num_classes = dataloader.dataset.num_classes
    warmup = True if epoch < args.warm_epochs else False

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    tau_clean = (args.tau_c_max - args.tau) * (epoch - args.warm_epochs) / (
            args.epochs - args.warm_epochs) + args.tau
    
    for i, (x, targets, idx) in enumerate(dataloader):
        loss_dict = {}
        batch_size = len(targets)
        x1 = x[0].cuda(non_blocking=True)

        given_labels = label_smoothing(targets, num_classes, args.eps)

        if warmup:
            loss = warmup_model(model, x1, given_labels)
            loss_dict.update({'loss_cls': loss})
        else:
            x2 = x[1].cuda(non_blocking=True)

            # normal logits & features
            feat = model(torch.cat([x1, x2], 0), return_feat=True)
            feat1, _ = feat.chunk(2)

            logits = model.classifier(feat)
            logits1, logits2 = logits.chunk(2)

            prob = F.softmax(logits, dim=1)
            prob1, prob2 = prob.chunk(2)

            # EMAs logits & features
            with torch.no_grad():
                feat_ema = ema_model(x1, return_feat=True)
                logits_ema = ema_model.module.classifier(feat_ema)


            ood_mask, clean_mask, id_noise_mask, high_conf_mask = (
                gen_mask(args, logits1, logits_ema, prob1, given_labels, tau_clean))


            loss_ood = cal_loss_ood(ood_mask, logits1, num_classes)
            loss_dict.update({'loss_ood': loss_ood * args.w_ood})

            losses = []
            if clean_mask.sum() > 0:
                queue.enqueue(feat1[clean_mask], targets[clean_mask.cpu()])
                loss_c = cal_loss_cls(logits1[clean_mask], logits2[clean_mask], given_labels[clean_mask], criterion)
                losses.append(loss_c)

            if high_conf_mask.sum() > 0:
                pse_labels, sem_labels, ema_labels = (
                    gen_pseudo_label(feat1, logits_ema, high_conf_mask, queue, args.t_proto, args.w_id))
                loss_id = cal_loss_cls(logits1[high_conf_mask], logits2[high_conf_mask], pse_labels, criterion)
                losses.append(loss_id)


            if len(losses) > 0:
                loss_cls = torch.cat(losses).mean()
                loss_dict.update({'loss_cls': loss_cls})

            loss_con = consistency_reg(prob1, prob2, ood_mask)
            if loss_con > 0:
                loss_dict.update({'loss_con': args.w_con * loss_con})

        loss = sum(loss_dict.values())
        loss_meter.update(loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema_model.update_parameters(model)

        if (i + 1) % 100 == 0:
            for k, v in loss_dict.items():
                if v > 0:
                    logger.info(f"{k}: {v.item():.4f}")
            logger.info(f"Epoch[{epoch}][{i}/{len(dataloader)}] Loss {loss_meter.val:.4f}({loss_meter})")

    logger.info(f'Train Summary Time {time.time() - end:.2f}s\t {loss_meter}\t tau_clean {tau_clean:.3f}')

