import torch.optim.lr_scheduler as lr_scheduler


def getScheduler(optimizer, opt):
    def lambda1(epoch): return 1.0 - max(0, epoch-opt.niter) / \
        float(opt.niter_decay + 1)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
