'''Train CIFAR10 with PyTorch.'''

import torch
import wandb
from time import time
from collections import deque
from sgd import sgd

from utils import AverageMeter, ProgressMeter


def transfer_gradients(net_1, net_2):
    for name, param in net_1.named_parameters():
        if net_2.get_parameter(name).grad is None:
            net_2.get_parameter(name).grad = param.grad.clone()
        else:
            net_2.get_parameter(name).grad += param.grad.clone()


def check_param_equality(model, parameters, gradients, momentum_buffers):
    for k, p in enumerate(net.parameters()):
        if not torch.allclose(p.data, parameters[k].data):
            raise ValueError(f'Parameter {k} is not updated properly')
        if not torch.allclose(p.grad.data, gradients[k].data):
            raise ValueError(f'Gradient {k} is not computed properly')
        if not torch.allclose(p.momentum_buf.data, momentum_buffers[k].data):
            raise ValueError(f'Momentum buffer {k} is not computed properly')


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


def l2_regularization_from_loss(model, device):
    l2_norm = torch.tensor(0.0, device=device)
    for k, p in model.named_parameters():
        if p.requires_grad:
            l2_norm += p.norm() ** 2
    return 5e-4 * l2_norm / 2.0


def l2_regularization_from_weights(model):
    for p in model.parameters():
        if p.requires_grad:
            p.grad += p * 5e-4


def init_training_delay(dataloader, model, criterion, optimizer, delay, decay_mode, decay_delayed):
    model.train()
    state_dict_queue = deque()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if optimizer is not None:
            optimizer.zero_grad()
        state_dict_queue.appendleft({k: v.clone() for k, v in model.state_dict().items()})
        if batch_idx >= delay:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # weight decay and backward pass
        if decay_mode == 'pytorch':
            loss.backward()
        elif decay_mode == 'loss':
            loss.backward()
            l2_regularization_from_loss(model, device).backward()
        elif decay_mode == 'weights':
            loss.backward()
            l2_regularization_from_weights(model)

        if optimizer is not None:
            optimizer.step()
        else:
            with torch.no_grad():
                if decay_mode == 'pytorch':
                    if decay_delayed:
                        raise ValueError('Delayed decay is not supported with pytorch decay mode.')
                    else:
                        params, grads, momentums = sgd(params=[p for p in model.parameters()],
                            d_p_list=[p.grad for p in model.parameters()],
                            momentum_buffer_list=[p.momentum_buf for p in model.parameters()],
                            lr=model.learning_rate,
                            momentum=model.momentum,
                            dampening=0.0,
                            weight_decay=model.weight_decay,
                            nesterov=False, maximize=False)
                        check_param_equality(model, params, grads, momentums)
                elif decay_mode in ['loss', 'weights']:
                    params, grads, momentums = sgd(params=[p for p in model.parameters()],
                        d_p_list=[p.grad for p in model.parameters()],
                        momentum_buffer_list=[p.momentum_buf for p in model.parameters()],
                        lr=model.learning_rate,
                        momentum=model.momentum,
                        dampening=0.0,
                        weight_decay=0.0,
                        nesterov=False, maximize=False)
                    check_param_equality(model, params, grads, momentums)

    return state_dict_queue


def train(dataloader, model, model_, criterion, optimizer, epoch, decay_mode, decay_delayed, use_wandb):
    if optimizer is None:
        print('\nEpoch: %d' % epoch, '- learning rate:', net.learning_rate)
    else:
        print('\nEpoch: %d' % epoch, '- learning rate:', optimizer.param_groups[0]['lr'])

    # put model in traininig
    model.train()
    model_.train()

    # trackers
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

        # load delayed weights and set grad to zero
        model.zero_grad()
        model_.load_state_dict(model_.state_stack.pop())
        model_.zero_grad()

        # forward pass
        outputs = model_(inputs)
        with torch.no_grad():
            _ = model(inputs)  # update running stats
        loss = criterion(outputs, targets)

        # weight decay and backward pass
        if decay_mode == 'pytorch':
            if decay_delayed:
                raise ValueError('Delayed decay is not supported with pytorch decay mode.')
            else:
                loss.backward()
                transfer_gradients(model_, model)
        elif decay_mode == 'loss':
            if decay_delayed:
                loss_regularized = loss + l2_regularization_from_loss(model_, device)
                loss_regularized.backward()
                transfer_gradients(model_, model)
            else:
                regularization = l2_regularization_from_loss(model, device)
                regularization.backward()

                loss.backward()
                transfer_gradients(model_, model)
        elif decay_mode == 'weights':
            if decay_delayed:
                loss.backward()
                l2_regularization_from_weights(model_)
                transfer_gradients(model_, model)
            else:
                loss.backward()
                transfer_gradients(model_, model)
                l2_regularization_from_weights(model)

        # update
        if optimizer is not None:
            optimizer.step()
        else:
            with torch.no_grad():
                if decay_mode == 'pytorch':
                    if decay_delayed:
                        raise ValueError('Delayed decay is not supported with pytorch decay mode.')
                    else:
                        params, grads, momentums = sgd(params=[p for p in model.parameters()],
                            d_p_list=[p.grad for p in model.parameters()],
                            momentum_buffer_list=[p.momentum_buf for p in model.parameters()],
                            lr=model.learning_rate,
                            momentum=model.momentum,
                            dampening=0.0,
                            weight_decay=model.weight_decay,
                            nesterov=False, maximize=False)
                        check_param_equality(model, params, grads, momentums)
                elif decay_mode in ['loss', 'weights']:
                    params, grads, momentums = sgd(params=[p for p in model.parameters()],
                        d_p_list=[p.grad for p in model.parameters()],
                        momentum_buffer_list=[p.momentum_buf for p in model.parameters()],
                        lr=model.learning_rate,
                        momentum=model.momentum,
                        dampening=0.0,
                        weight_decay=0.0,
                        nesterov=False, maximize=False)
                    check_param_equality(model, params, grads, momentums)

        # storing new model state
        model.zero_grad()
        model_.state_stack.appendleft({k: v.clone() for k, v in model.state_dict().items()})

        # track metrics
        batch_time.update(time() - end)
        losses.update(loss.item(), n=inputs.size(0))
        top1.update(accuracy(outputs, targets)[0].item(), n=inputs.size(0))
        if batch_idx % step == step - 1:
            progress.display(batch_idx + 1)
        end = time()

    # log metrics
    if use_wandb:
        wandb.log({'loss/train': losses.avg, 'acc/train': top1.avg})


def test(dataloader, model, criterion, epoch, use_wandb):
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
    # progress.display(batch_idx + 1)
    print(
        f'Test Epoch {epoch} - Data {data_time.avg: 6.3f} - Time {batch_time.avg: 6.3f} - Loss {losses.avg: .4e} - Acc {top1.avg: 6.2f}')
    # log metrics
    if use_wandb:
        wandb.log({'loss/test': losses.avg, 'acc/test': top1.avg})

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
    import os
    import argparse
    import torchvision
    import torchvision.transforms as transforms
    import torch.backends.cudnn as cudnn
    from models.resnet import ResNet18
    import torch.optim as optim

    start_time = time()

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--delay', default=0, type=int, help='delay')
    parser.add_argument('--decay-mode', type=str, default='pytorch', choices=['pytorch', 'loss', 'weights'])
    parser.add_argument('--decay-delayed', action='store_true', default=False)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['sgd-class', 'sgd-function'])
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr', 'onecycle'])
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-wandb', action='store_true', help='no wandb logging')
    args = parser.parse_args()

    args.wandb = not args.no_wandb
    del args.no_wandb

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
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'sgd-class':
        if args.decay_mode == 'pytorch':
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.decay_mode in ['loss', 'weights']:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        else:
            raise ValueError(f'Wrong decay mode ({args.decay_mode})')
    elif args.optimizer == 'sgd-function':
        net.learning_rate = args.lr
        net.momentum = 0.9
        net.weight_decay = 5e-4
        optimizer = None
        for p in net.parameters():
            p.momentum_buf = None

    # Scheduler
    if args.optimizer == 'sgd-class':
        if args.scheduler == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        elif args.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        else:
            raise ValueError(f'Wrong scheduler type ({args.scheduler})')

    # Init delay
    print('\nInitliazing delay: %d' % args.delay)
    net_.state_stack = init_training_delay(trainloader, net, criterion, optimizer, args.delay, args.decay_mode,
                                           args.decay_delayed)

    if args.wandb:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project='Delayed Gradients', entity='streethagore', config=args, group="uniform-delay")

    # training loop
    for epoch in range(start_epoch, start_epoch + 100):
        if args.optimizer == 'sgd-function' and epoch in [30, 60, 90]:
            net.learning_rate *= 0.2
        train(trainloader, net, net_, criterion, optimizer, epoch, args.decay_mode, args.decay_delayed, args.wandb)
        test(testloader, net, criterion, epoch, args.wandb)
        if args.optimizer == 'sgd-class':
            scheduler.step()

    if args.wandb:
        wandb.summary["duration"] = time() - start_time
        wandb.summary["best-accuracy"] = net.best_acc
        wandb.finish()
