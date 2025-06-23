jax
JAX version: --jax
Output only code.
</think>
import jax
import jax.numpy as jnp
from jax import vmap, grad
from jax.experimental import setup_custom_eols
from jax import app
import argparse
import os
import sys
import shutil
import time
from importlib import import_module
from train import train, val, test, Logger

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', default='sample', help='model')
    parser.add_argument('-j', '--workers', default=32, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, help='weight decay')
    parser.add_argument('--save-freq', default='1', type=int, help='save frequency')
    parser.add_argument('--print-iter', default=0, type=int, help='print per iter')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--save-dir', default='', type=str, help='directory to save checkpoint')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--test', default='0', type=int, help='test mode')
    parser.add_argument('--defense', default=1, type=int, help='test mode')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer')

    args = parser.parse_args()

    modelpath = os.path.join(os.path.abspath('../pixel_Exps'),args.exp)
    train_data = jnp.load(os.path.join(modelpath,'train_split.npy'))
    val_data = jnp.load(os.path.join(modelpath,'val_split.npy'))
    with open(os.path.join(modelpath,'train_attack.txt')) as f:
        train_attack = [line.split(' ')[0].split(',')[0] for line in f.readlines()]
    
    sys.path.append(modelpath)
    model = import_module('model')
    config, net = model.get_model()
    
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    if args.resume:
        checkpoint = jax.loadCheckpoint(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join(modelpath,'results', save_dir)
        net.update_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join(modelpath,'results', exp_id)
        else:
            save_dir = os.path.join(modelpath,'results', save_dir)
    
    if args.test == 1:
        with open(os.path.join(modelpath,'test_attack.txt')) as f:
            test_attack = [line.split(' ')[0].split(',')[0] for line in f.readlines()]
        test_data = jnp.load(os.path.join(modelpath,'test_split.npy'))
        dataset = DefenseDataset(config, 'test', test_data, test_attack)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        args.defense = args.defense == 1
        name = f'result_{args.exp}_{args.resume.split(".")[-1]}' if args.resume else f'result_{args.exp}'
        test(net, test_loader, name, args.defense)
        return

    dataset = DefenseDataset(config, 'train', train_data, train_attack)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        log_file = os.path.join(save_dir, 'log')
        sys.stdout = Logger(log_file)
        py_files = [f for f in os.listdir('.') if f.endswith('.py')]
        for f in py_files:
            shutil.copy(f, os.path.join(save_dir,