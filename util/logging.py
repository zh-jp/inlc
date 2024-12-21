import os
from loguru import logger

def create_logger(args, name=None):
    if os.path.exists('./checkpoint') is False:
        os.makedirs('./checkpoint')

    if args.dataset in [10, 100]:
        asym = '-asym' if args.asym else ''
        name = f'cifar{args.dataset}-{args.backbone}-{args.r_ood}-{args.r_id}-{args.r_imb}{asym}-{args.mark}'
    else:
        assert name is not None, 'Please provide a name for the logger'

    args.name = name
    logger.add(f'./log/{name}/{{time}}.log')

    for k, v in vars(args).items():
        if v is not None:
            logger.info(f'{k}: {v}')

    return logger
