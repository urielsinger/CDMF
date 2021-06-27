import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='random', choices=['random'], help='name of the dataset')
parser.add_argument('--max_seq_len', type=int, default=1000, help='maximum length of a sequence')
parser.add_argument('--per_user', action="store_true", default=False, help='learn different behavior per user')
parser.add_argument('--tau', type=float, default=0.01, help='used for activation function: max(x,tau)')
parser.add_argument('--emb_dim', type=int, default=128, help='item embedding size')

parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=128, help='optimizer weight decay')
parser.add_argument('--max_epochs', type=int, default=5, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=32, help='number of samples in batch')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
parser.add_argument('--gpus', type=str, default='0', help='gpus parameter used for pytorch_lightning')
parser.add_argument('--seed', type=int, default=43, help='random seed')
