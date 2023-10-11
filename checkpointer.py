import torch
import os

def save_checkpoint(epoch, model, optimizer, ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(ckpt_dir, 'checkpoint_%04d.pt' % epoch))


def load_checkpoint(ckpt_dir):
    """
    start_epoch = 0

    checkpoint = load_checkpoint(self.cfg.paths.ckpt_in_dir)
    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        scheduler.last_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for g in optimizer.param_groups:
            g['lr'] = self.cfg.optim.base_lr
        model.load_state_dict(checkpoint['model_state_dict'])
    """
    if not os.path.isdir(ckpt_dir):
        print('Checkpoint directory not found. No checkpoints loaded from ', ckpt_dir)
        return None

    files = os.listdir(ckpt_dir)
    files = list(filter(lambda x: x.startswith('checkpoint'), files))

    if len(files) == 0:
        print('No checkpoint found in ', ckpt_dir)
        return None

    files = sorted(files, reverse=True)
    latest_fp = os.path.join(ckpt_dir, files[0])
    print('Loading checkpoint from ' + latest_fp)

    return torch.load(latest_fp)