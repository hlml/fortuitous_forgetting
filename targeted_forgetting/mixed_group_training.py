import argparse
import copy
import os
import os.path
import pickle
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
import wandb
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from torchvision.datasets.vision import VisionDataset
from urllib.error import URLError
from torchvision.datasets.mnist import *
ACT = F.relu

data_path = '/home/datasets'
use_cuda = torch.cuda.is_available()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_iters', type=int, default=2000, help='number of training iterations')
    parser.add_argument('--train_perc', type=float, default=1.0, help='percentage of data for training')
    parser.add_argument('--random_perc', type=float, default=0.0, 
            help='percentage of data with random labels')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--save_path', type=str, default='exp_logged', help='save model object to this path')
    parser.add_argument('--fname', type=str, default='base', help='group name for wandb')
    parser.add_argument('--print_every', type=int, default=100, help='print status update every n iterations')
    parser.add_argument('--resume_path', type=str, default=None, help='load a pretrained model from this path')
    parser.add_argument('--rewind_to_init', action='store_true', help='rewind to init for fit hypothesis')
    parser.add_argument('--fixed_mask', action='store_true', help='use the same mask every generation')
    parser.add_argument('--train_data', type=str, default='mnist', choices=('mnist', 'cifar10'),
                        help='name of training dataset')
    parser.add_argument('--arch', type=str, default='lenet', choices=('lenet', 'conv4', 'resnet18'),
                        help='model architecture choices. Use lenet for mnist, conv4 and resnet18 for cifar10')
    parser.add_argument('--same_mask', action='store_true')
    parser.add_argument('--weight_mask', action='store_true')
    parser.add_argument('--reset_to_zero', action='store_true')
    parser.add_argument('--margin_groups', action='store_true', help="true if splitting group based on output margin, false if using random label splits")
    parser.add_argument('--keep_perc', type=float, default=1)
    parser.add_argument("--group_vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='do not write to wandb')

    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam'))
    parser.add_argument('--rewind_to_epoch', type=int, default=0,
                        help='rewind to epoch n and start tracking from this point')
    parser.add_argument('--num_workers', type=int, default=6)
    return parser


def load_datasets(data_path):
    d_tr, d_te = torch.load(data_path)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Simple Conv Block
class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool=True):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.conv = nn.Conv2d(indim, outdim, 3, padding=1)
        #         self.bn     = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        #         self.parametrized_layers = [self.conv, self.bn, self.relu]
        self.parametrized_layers = [self.conv, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out

def init_layer(L):
    # Initialization using glorot normal
    if isinstance(L, nn.Conv2d):
        torch.nn.init.kaiming_normal_(L.weight)
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# for cifar10
class ConvNet4(nn.Module):
    def __init__(self, flatten=True):
        super(ConvNet4, self).__init__()

        A = ConvBlock(3, 64, pool=False)
        B = ConvBlock(64, 64, pool=True)
        C = ConvBlock(64, 128, pool=False)
        D = ConvBlock(128, 128, pool=True)
        trunk = [A, B, C, D]

        if flatten:
            trunk.append(Flatten())

        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

        trunk.append(self.fc1)
        trunk.append(nn.ReLU(inplace=True))
        trunk.append(self.fc2)
        trunk.append(nn.ReLU(inplace=True))
        trunk.append(self.fc3)

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


class CifarResNet18(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(CifarResNet18.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return self.relu2(out)

    def __init__(self):
        super(CifarResNet18, self).__init__()

        def kaiming_normal(w):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(w.weight)

        outputs = 10
        plan = [(16, 3), (2 * 16, 3), (4 * 16, 3)]
        initializer = kaiming_normal
        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(CifarResNet18.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CustomCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            random_perc: float = 0.0,
            download: bool = False,
    ) -> None:

        super(CustomCIFAR10, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        self.train = train  # training set or test set
        self.random_perc = random_perc

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.rand_group_ids = (np.random.random(self.data.shape[0]) < self.random_perc).astype(int)

        if self.random_perc > 0:
            temp_targets = np.array(self.targets)
            temp_targets[self.rand_group_ids == 1] = np.random.choice(10, self.rand_group_ids.sum())
            self.targets = list(temp_targets)

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, rand_group_id = self.data[index], self.targets[index], self.rand_group_ids[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, rand_group_id

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CustomMNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            random_perc: float = 0.0,
            download: bool = False,
    ) -> None:
        super(CustomMNIST, self).__init__(root, transform=transform,
                                          target_transform=target_transform)
        self.train = train  # training set or test set
        self.random_perc = random_perc

        if self._check_legacy_exist():
            self.data, self.targets, self.rand_group_ids = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets, self.rand_group_ids = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        data, targets = torch.load(os.path.join(self.processed_folder, data_file))
        rand_group_ids = (np.random.random(targets.shape[0]) < self.random_perc).astype(int)

        if self.random_perc > 0:
            targets[rand_group_ids == 1] = torch.tensor(np.random.choice(10, rand_group_ids.sum()))

        return data, targets, rand_group_ids

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        rand_group_ids = (np.random.random(targets.shape[0]) < self.random_perc).astype(int)

        if self.random_perc > 0 and self.train:
            targets[rand_group_ids == 1] = torch.tensor(np.random.choice(10, rand_group_ids.sum()))

        return data, targets, rand_group_ids

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, rand_group_id = self.data[index], int(self.targets[index]), int(self.rand_group_ids[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, rand_group_id

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
    

def test(loader, model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct, total, total_loss = 0, 0, 0
    tot_iters = len(loader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs, targets = next(iter(loader))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)

            _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            total_loss += criterion(outputs, targets).item()

    # Save checkpoint.
    acc = 100. * float(correct) / float(total)
    loss = total_loss / tot_iters
    return loss, acc


def torch_bernoulli(p, size):
    return torch.rand(size) < p


def main():
    parser = make_parser()
    args = parser.parse_args()

    # set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if not args.no_wandb:
        if len(args.group_vars) > 0:
            if len(args.group_vars) == 1:
                group_name = args.group_vars[0] + str(getattr(args, args.group_vars[0]))
            else:
                group_name = args.group_vars[0] + str(getattr(args, args.group_vars[0]))
                for var in args.group_vars[1:]:
                    group_name = group_name + '_' + var + str(getattr(args, var))
            wandb.init(project="mixed_group_training",
                       group=args.fname,
                       name=group_name)
            for var in args.group_vars:
                wandb.config.update({var: getattr(args, var)})

    if args.train_data == 'mnist':
        trans = ([transforms.ToTensor()])
        trans = transforms.Compose(trans)

        dset = CustomMNIST(root=data_path, train=True, download=True, transform=trans, random_perc=args.random_perc)
        train_set, _ = torch.utils.data.random_split(dset, [int(60000 * args.train_perc),
                                                            60000 - int(60000 * args.train_perc)])
        test_set = CustomMNIST(root=data_path, train=False, download=True, transform=trans,
                               random_perc=args.random_perc)


    elif args.train_data == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        test_transform = transforms.Compose(
            [transforms.ToTensor()])

        dset = CustomCIFAR10(root=data_path, train=True, download=True, transform=train_transform,
                             random_perc=args.random_perc)
        train_set, _ = torch.utils.data.random_split(dset, [int(50000 * args.train_perc),
                                                            50000 - int(50000 * args.train_perc)])
        test_set = CustomCIFAR10(root=data_path, train=False, download=True, transform=test_transform,
                                 random_perc=args.random_perc)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=args.num_workers)

    if args.arch == 'lenet':
        model = LeNet()
        modelClone = LeNet()
    elif args.arch == 'conv4' and args.train_data == 'cifar10':
        model = ConvNet4()
        modelClone = ConvNet4()
    elif args.arch == 'resnet18' and args.train_data == 'cifar10':
        model = CifarResNet18()
        modelClone = CifarResNet18()

    if args.resume_path is not None:
        model_checkpoint = torch.load(args.resume_path)
        model.load_state_dict(model_checkpoint['state_dict'])

    if use_cuda:
        model.cuda()
        modelClone.cuda()

    if args.rewind_to_init:
        init_weights = {}
        for name, param in model.named_parameters():
            init_weights[name] = copy.deepcopy(param.data.detach().clone())

    if args.same_mask:
        if args.weight_mask:
            mask_dict = {}
            for name, param in model.named_parameters():
                if 'bn' not in name and 'shortcut.1' not in name:
                    weight_mag = torch.abs(param.detach().clone())
                    topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement() * (1 - args.keep_perc)),
                                      largest=False)
                    temp_mask = torch.ones(weight_mag.nelement())
                    temp_mask[topk.indices] = 0
                    mask_dict[name] = temp_mask.bool().view(weight_mag.shape)
        else:
            mask_dict = {}
            for name, param in model.named_parameters():
                if 'bn' not in name and 'shortcut.1' not in name:
                    mask_dict[name] = torch_bernoulli(args.keep_perc, param.shape)

    criterion = nn.CrossEntropyLoss()
    if args.train_data == 'mnist':
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    else:
        if args.arch == 'conv4':
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

            def assign_learning_rate(optimizer, new_lr):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

    if args.rewind_to_epoch > 0:
        orig_print_every = args.print_every
        args.print_every = args.train_iters
        args.rewind_to_init = False

    true_entropy = 0
    rand_entropy = 0
    iter_count = 0
    for ep in range(9999):

        if args.rewind_to_epoch > 0 and ep == args.rewind_to_epoch:
            args.print_every = orig_print_every
            args.rewind_to_init = True

            init_weights = {}
            for name, param in model.named_parameters():
                init_weights[name] = copy.deepcopy(param.data.detach().clone())

        if args.train_data == 'cifar10' and args.arch == 'resnet18':
            if ep == 50:
                assign_learning_rate(optimizer, 0.01)
            if ep == 100:
                assign_learning_rate(optimizer, 0.001)

        for i, data in enumerate(trainloader):
            if iter_count < args.train_iters:
                model.train()
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, group_ids = data

                if inputs.size(0) < 128:
                    continue

                if use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    group_ids = group_ids.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # evaluate training accuracy
                iter_count += 1
                if iter_count % args.print_every == 0:
                    modelClone.load_state_dict(model.state_dict())

                    with torch.no_grad():
                        if args.margin_groups:
                            logits = nn.Softmax(dim=1)(outputs)  # (128,10)
                            # get hard examples
                            logits_conf = []
                            logits_num = logits.cpu().numpy()
                            for i in range(logits_num.shape[0]):
                                sort_id = np.argsort(-logits_num[i, :])
                                logits_conf.append(logits_num[i, sort_id[0]] - logits_num[i, sort_id[1]])
                            hard_idx = np.argsort(logits_conf)[:int(len(logits_conf) * 0.1)]
                            group_ids = torch.zeros(len(logits_conf))
                            group_ids[hard_idx] = 1

                        _, predicted = torch.max(nn.Softmax(dim=1)(outputs), 1)
                        correct = predicted.eq(labels.data).cpu()
                        true_acc = correct[group_ids == 0].float().mean()
                        rand_acc = correct[group_ids == 1].float().mean()
                        overall_acc = correct.float().mean()

                        true_loss = criterion(outputs[group_ids == 0], labels[group_ids == 0])
                        rand_loss = criterion(outputs[group_ids == 1], labels[group_ids == 1])


                        logits = nn.Softmax(dim=1)(outputs)  # (128,10)
                        true_entropy = torch.mean(
                            torch.sum(torch.log(logits[group_ids == 0, :]) * -logits[group_ids == 0, :],
                                      1)).cpu().numpy()
                        rand_entropy = torch.mean(
                            torch.sum(torch.log(logits[group_ids == 1, :]) * -logits[group_ids == 1, :],
                                      1)).cpu().numpy()

                    if args.rewind_to_init:
                        for name, param in modelClone.named_parameters():
                            param.data = copy.deepcopy(init_weights[name])

                    if not args.same_mask:
                        if args.weight_mask:
                            mask_dict = {}
                            for name, param in model.named_parameters():
                                if 'bn' not in name and 'bias' not in name and 'shortcut.1' not in name:
                                    weight_mag = torch.abs(param.detach().clone())
                                    if args.arch == 'resnet18':
                                        if 'fc' not in name:
                                            prune_rate = 1 - args.keep_perc
                                        else:
                                            prune_rate = (1 - args.keep_perc) / 2
                                    elif args.arch == 'conv4':
                                        if 'fc' in name:
                                            prune_rate = 1 - args.keep_perc
                                        else:
                                            prune_rate = (1 - args.keep_perc) / 2
                                    elif args.arch == 'lenet':
                                        if 'fc3' not in name:
                                            prune_rate = 1 - args.keep_perc
                                        else:
                                            prune_rate = (1 - args.keep_perc) / 2

                                    topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement() * prune_rate),
                                                      largest=False)

                                    temp_mask = torch.ones(weight_mag.nelement())
                                    temp_mask[topk.indices] = 0
                                    mask_dict[name] = temp_mask.bool().view(weight_mag.shape)
                        else:
                            mask_dict = {}
                            for name, param in model.named_parameters():
                                if 'bn' not in name and 'bias' not in name and 'shortcut.1' not in name:
                                    if args.arch == 'resnet18':
                                        if 'fc' not in name:
                                            keep_rate = args.keep_perc  # for conv layers, keep 50%
                                        else:
                                            keep_rate = args.keep_perc + (
                                                        (1 - args.keep_perc) / 2)  # for fc layers, keep 75%
                                    elif args.arch == 'conv4':
                                        if 'fc' in name:
                                            keep_rate = args.keep_perc  # for fc layers, keep 50%
                                        else:
                                            keep_rate = args.keep_perc + (
                                                        (1 - args.keep_perc) / 2)  # for conv layers, keep 75%
                                    elif args.arch == 'lenet':
                                        if 'fc3' not in name:
                                            keep_rate = args.keep_perc  # for first two layers, keep 50%
                                        else:
                                            keep_rate = args.keep_perc + (
                                                        (1 - args.keep_perc) / 2)  # for last layer, keep 75%

                                    mask_dict[name] = torch_bernoulli(keep_rate, param.shape)

                    if args.reset_to_zero:
                        for name, param in modelClone.named_parameters():
                            if 'bn' not in name and 'bias' not in name and 'shortcut.1' not in name:
                                param.data *= mask_dict[name].float().to('cuda')
                    else:
                        metric_dict = {}
                        for name, param in modelClone.named_parameters():
                            if 'bn' not in name and 'bias' not in name and 'shortcut.1' not in name:
                                metric_dict[name] = param.detach().clone()

                        for name, param in modelClone.named_children():
                            if hasattr(param, 'reset_parameters') and 'bn' not in name and 'shortcut.1' not in name:
                                param.reset_parameters()

                        for name, param in modelClone.named_parameters():
                            if 'bn' not in name and 'bias' not in name and 'shortcut.1' not in name:
                                param.data[mask_dict[name]] = metric_dict[name][mask_dict[name]]

                    # calculate copy accuracy
                    with torch.no_grad():
                        copy_outputs = modelClone(inputs)
                        copy_loss = criterion(copy_outputs, labels)

                        # evaluate training accuracy
                        _, copy_predicted = torch.max(nn.Softmax(dim=1)(copy_outputs), 1)
                        copy_correct = copy_predicted.eq(labels.data).cpu()
                        copy_true_acc = copy_correct[group_ids == 0].float().mean()
                        copy_rand_acc = copy_correct[group_ids == 1].float().mean()
                        copy_overall_acc = copy_correct.float().mean()

                        copy_true_loss = criterion(copy_outputs[group_ids == 0], labels[group_ids == 0])
                        copy_rand_loss = criterion(copy_outputs[group_ids == 1], labels[group_ids == 1])

                        copy_logits = nn.Softmax(dim=1)(copy_outputs)
                        copy_true_entropy = torch.mean(
                            torch.sum(torch.log(copy_logits[group_ids == 0, :]) * -copy_logits[group_ids == 0, :],
                                      1)).cpu().numpy()
                        copy_rand_entropy = torch.mean(
                            torch.sum(torch.log(copy_logits[group_ids == 1, :]) * -copy_logits[group_ids == 1, :],
                                      1)).cpu().numpy()

                    if args.no_wandb:
                        record = 'Iteration ' + str(iter_count) \
                                 + ' Train loss ' + str(np.round(loss.item(), decimals=4)) \
                                 + ' True Label Accuracy ' + str(np.round(true_acc * 100, decimals=2)) \
                                 + ' Random Label Accuracy ' + str(np.round(rand_acc * 100, decimals=2)) \
                                 + ' Overall Accuracy ' + str(np.round(overall_acc * 100, decimals=2)) \
                                 + ' Copy Train loss ' + str(np.round(copy_loss.item(), decimals=4)) \
                                 + ' Copy True Label Accuracy ' + str(np.round(copy_true_acc * 100, decimals=2)) \
                                 + ' Copy Random Label Accuracy ' + str(np.round(copy_rand_acc * 100, decimals=2)) \
                                 + ' Copy Overall Accuracy ' + str(np.round(copy_overall_acc * 100, decimals=2)) + '%\n'
                        print(record)
                    else:
                        wandb.log({'Iter': iter_count, 'Train loss': np.round(loss.item(), decimals=4),
                                   'True Label Accuracy': np.round(true_acc * 100, decimals=2),
                                   'Random Label Accuracy': np.round(rand_acc * 100, decimals=2),
                                   'Overall Accuracy': np.round(overall_acc * 100, decimals=2),
                                   'Copy Train loss': np.round(copy_loss.item(), decimals=4),
                                   'Copy True Label Accuracy': np.round(copy_true_acc * 100, decimals=2),
                                   'Copy Random Label Accuracy': np.round(copy_rand_acc * 100, decimals=2),
                                   'Copy Overall Accuracy': np.round(copy_overall_acc * 100, decimals=2),
                                   'Copy True Entropy': np.round(copy_true_entropy.item(), decimals=4),
                                   'Copy Random Entropy': np.round(copy_rand_entropy.item(), decimals=4),
                                   'True Entropy': np.round(true_entropy.item(), decimals=4),
                                   'Random Entropy': np.round(rand_entropy.item(), decimals=4),
                                   'True loss': np.round(true_loss.item(), decimals=4),
                                   'Random loss': np.round(rand_loss.item(), decimals=4),
                                   'Copy True loss': np.round(copy_true_loss.item(), decimals=4),
                                   'Copy Random loss': np.round(copy_rand_loss.item(), decimals=4)})


            else:
                if not args.no_wandb:
                    wandb.run.finish()
                return 'Done'


if __name__ == '__main__':
    main()
