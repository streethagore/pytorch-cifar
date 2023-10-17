'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import AverageMeter, ProgressMeter
from collections import deque
from time import time


def transfer_gradients(net_1, net_2):
    for name, param in net_1.named_parameters():
        net_2.get_parameter(name).grad = param.grad.clone()


def init_training_delay(dataloader, model, criterion, optimizer, delay):
    model.train()
    state_dict_queue = deque()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        state_dict_queue.appendleft({k: v.clone() for k, v in model.state_dict().items()})
        if batch_idx >= delay:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return state_dict_queue


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(dataloader, model, model_, criterion, optimizer, epoch, sync_p):
    print('\nEpoch: %d' % epoch)
    model.train()
    model_.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1],
                             prefix="Train: [{}]".format(epoch))
    step = max(len(dataloader) // 10, 1)
    end = time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        data_time.update(time() - end)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if sync_p > 0 and batch_idx % sync_p == sync_p - 1:
            model_.state_stack = init_training_delay(trainloader, net, criterion, optimizer, args.delay)
        model_.load_state_dict(model_.state_stack.pop())
        model_.zero_grad()

        outputs = model_(inputs)
        with torch.no_grad():
            _ = model(inputs)  # update running stats
        loss = criterion(outputs, targets)
        loss.backward()
        transfer_gradients(model_, model)
        optimizer.step()

        model_.state_stack.appendleft({k: v.clone() for k, v in model.state_dict().items()})
        batch_time.update(time() - end)

        losses.update(loss.item(), n=inputs.size(0))
        top1.update(accuracy(outputs, targets)[0].item(), n=inputs.size(0))
        if batch_idx % step == step - 1:
            progress.display(batch_idx + 1)
        end = time()


def test(dataloader, model, criterion, epoch):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1],
                             prefix="Test: [{}]".format(epoch))
    end = time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            data_time.update(time() - end)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss, n=inputs.size(0))
            top1.update(accuracy(outputs, targets)[0].item(), n=inputs.size(0))
            batch_time.update(time() - end, n=inputs.size(0))
            end = time()
    progress.display(batch_idx + 1)
    print(f'Test Epoch {epoch} - Data {data_time.avg: 6.3f} - Time {batch_time.avg: 6.3f} - Loss {losses.avg: .4e} - Acc {top1.avg: 6.2f}')

    # Save checkpoint.
    acc = 100. * top1.avg
    if acc > model.best_acc:
        # print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        model.best_acc = acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--delay', default=0, type=int, help='delay')
    parser.add_argument('--sync-p', default=0, type=int, help='synchronization period')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else torch.device('mps')

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    net_ = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()

    net.to(device)
    net_.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    net.best_acc = 0.0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        net.best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Init delay
    print('\nInitliazing delay: %d' % args.delay)
    net_.state_stack = init_training_delay(trainloader, net, criterion, optimizer, args.delay)
    for epoch in range(start_epoch, start_epoch + 100):
        train(trainloader, net, net_, criterion, optimizer, epoch, args.sync_p)
        test(testloader, net, criterion, epoch)
        scheduler.step()
