import collections
import os
import random

from torchvision import datasets


def dump_split(split_path, data, members):
    print(split_path)
    for cls, indices in members.items():
        print(cls)
        os.makedirs(split_path + "/" + str(cls))
        for idx in indices:
            img, label = data[idx]
            assert cls == label
            img.save(split_path + "/" + str(cls) + "/" + str(idx) + ".png")


def dump_data(root_path, train_data, train_members, test_data, test_members):
    dump_split(root_path + "/train", train_data, train_members)
    dump_split(root_path + "/test", test_data, test_members)


if __name__ == "__main__":
    split_frac = 0.85
    train_data = datasets.CIFAR100(root='./cifar100_raw', train=True, download=True)
    test_data = datasets.CIFAR100(root='./cifar100_raw', train=False, download=True) 

    train_members = collections.defaultdict(list)
    test_members = collections.defaultdict(list)
    for i, (_, label) in enumerate(train_data):
        train_members[label].append(i)
    for i, (_, label) in enumerate(test_data):
        test_members[label].append(i)

    subtrain_members = {}
    subval_members = {}
    for k, v in train_members.items():
        random.shuffle(v)
        subtrain_members[k] = v[:int(len(v) * split_frac)]
        subval_members[k] = v[int(len(v) * split_frac):]

    dump_data('./CIFAR100', train_data, train_members, test_data, test_members)
    dump_data('./CIFAR100val', train_data, subtrain_members, train_data, subval_members)