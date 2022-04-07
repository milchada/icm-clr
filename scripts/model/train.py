import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from torchsummary import summary
from scripts.data.SimClrDataset import SimClrDataset
from scripts.data.transforms import get_transforms
from scripts.model.resnet_simclr import ResNetSimCLR
from scripts.model.simclr import SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='~/simclr/dataset/m_train.csv',
                    help='path to training dataset')
parser.add_argument('--val_data', metavar='DIR', default='~/simclr/dataset/m_val.csv',
                    help='path to validation dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


parser.add_argument('--representation_dim', default=128, type=int,
                    help='representation dimension (default: 128)')

parser.add_argument('--projection_dim', default=128, type=int,
                    help='projection dimension (default: 128)')

parser.add_argument('--depth', default=16, type=int,
                    help='resnet depth (default: 16)')

parser.add_argument('--widen_factor', default=1, type=int,
                    help='widen factor of the resnet (default: 8)')

parser.add_argument('--dropout_rate', default=0.3, type=float,
                    help='dropout rate of the resnet (default: 0.3)')

parser.add_argument('--num_channels', default=3, type=int,
                    help='resnet depth dimension (default: 3)')

parser.add_argument('--image_size', default=128, type=int,
                    help='image size in pixel per side (default: 128)')

parser.add_argument('--patience', default=10, type=int,
                    help='stop after loss has not improved for this amount of epochs (default: 10)')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

        
    transforms = get_transforms(args.image_size)    
    train_dataset = SimClrDataset(args.data, transform=transforms, n_views=args.n_views)
    val_dataset = SimClrDataset(args.val_data, transform=transforms, n_views=args.n_views)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(args.depth,
                         args.widen_factor,
                         args.dropout_rate,
                         args.representation_dim,
                         args.projection_dim,
                         args.num_channels)
    
    summary(model.cuda(), (3, 128, 128))

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, val_loader)


if __name__ == "__main__":
    main()