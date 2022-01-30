import argparse


def parse():
    parser = argparse.ArgumentParser(description='Referential game settings')

    parser.add_argument('--gpu', type=int, default=0, help='which gpu if we use gpu')
    parser.add_argument('--fname', type=str, default='test', help='folder name to save results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--jupyter', action='store_true') 
    parser.add_argument('--slambda', type=float, default=0.1, help='speaker regularization hyperparameter')
    parser.add_argument('--rlambda', type=float, default=0.1, help='listener regularization hyperparameter')
    parser.add_argument('--receiverNum', type=int, default=1, help='number of listeners in the population')
    parser.add_argument('--topk', type=int, default=3, help='number of top messages when we probe language')
    parser.add_argument('--evaluateSize', type=int, default=1000, help='the batch size of test objects when not enumeration')
    args_dict = vars(parser.parse_args()) # convert python object to dict
    return args_dict
