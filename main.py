import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from visdom import Visdom
viz = Visdom()

import data
import model

from dnc.util import register_nan_checks

from utils import batchify, get_batch, repackage_hidden, repackage_hidden_dnc

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, DNC, SDNC)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--nhlayers', type=int, default=1,
                    help='number of hidden layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--optim', type=str, default='adam', \
                    help='learning rule, supports adam|rmsprop')
parser.add_argument('--reset', action='store_true',
                    help='Reset DNC memory contents on every forward pass')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--debug', action='store_true',
                    help='debug DNC memory contents in visdom (on localhost)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--start-decay-at', type=int, default=50,
                    help='start decaying learning rate from this epoch')
parser.add_argument('--decay-multiplier', type=float, default=0.5,
                    help='this will be multiplied with the current learning \
                    rate every epoch starting from --start-decay-at epochs')


parser.add_argument('--nr_cells', type=int, default=8, help='Number of memory cells of the DNC / SDNC')
parser.add_argument('--read_heads', type=int, default=4, help='Number of read heads of the DNC / SDNC')
parser.add_argument('--sparse_reads', type=int, default=4, help='Number of sparse memory cells recalled per read head for SDNC')
parser.add_argument('--cell_size', type=int, default=400, help='Cell sizes of DNC / SDNC')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.cuda == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(
    args.model,
    ntokens,
    args.emsize,
    args.nhid,
    args.nlayers,
    args.nhlayers,
    args.dropout,
    args.dropouth,
    args.dropouti,
    args.dropoute,
    args.wdrop,
    False,
    args.nr_cells,
    args.read_heads,
    args.sparse_reads,
    args.cell_size,
    args.cuda
)
register_nan_checks(model)
if args.cuda != -1:
    model.cuda(args.cuda)
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden, _ = model(data, hidden, reset_experience=True)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        if 'dnc' not in args.model.lower():
            hidden = repackage_hidden(hidden)
        else:
            hidden = repackage_hidden_dnc(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    debug_mem = None
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if 'dnc' not in args.model.lower():
            hidden = repackage_hidden(hidden)
        else:
            hidden = repackage_hidden_dnc(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs, debug_mem = model(data, hidden, return_h=True, reset_experience=args.reset)
        raw_loss = criterion(output.view(-1, ntokens), targets)

        loss = raw_loss
        # Activiation Regularization
        loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            try:
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.8f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            except Exception as e:
                print('Exception in debug')
                pass
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    return debug_mem

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:

    for epoch in range(1, args.epochs+1):

        if args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
        if args.optim == 'sparseadam':
            optimizer = optim.SparseAdam(model.parameters(), lr=lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
        if args.optim == 'adamax':
            optimizer = optim.Adamax(model.parameters(), lr=lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
        elif args.optim == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1e-10) # 0.0001
        elif args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=args.wdecay) # 0.01
        elif args.optim == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        elif args.optim == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=lr)

        epoch_start_time = time.time()
        v = train()

        val_loss2 = evaluate(val_data)

        if False:
            viz.heatmap(
                v[0]['memory'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Memory, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                    ylabel='layer * time',
                    xlabel='mem_slot * mem_size'
                )
            )

            viz.heatmap(
                v[0]['link_matrix'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Link Matrix, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                    ylabel='layer * time',
                    xlabel='mem_slot * mem_slot'
                )
            )

            viz.heatmap(
                v[0]['precedence'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Precedence, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                    ylabel='layer * time',
                    xlabel='mem_slot'
                )
            )

            viz.heatmap(
                v[0]['read_weights'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Read Weights, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                    ylabel='layer * time',
                    xlabel='nr_read_heads * mem_slot'
                )
            )

            viz.heatmap(
                v[0]['write_weights'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Write Weights, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                    ylabel='layer * time',
                    xlabel='mem_slot'
                )
            )

            viz.heatmap(
                v[0]['usage_vector'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Usage Vector, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                    ylabel='layer * time',
                    xlabel='mem_slot'
                )
            )

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss2, math.exp(val_loss2)))
        print('-' * 89)

        if val_loss2 < stored_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            print('Saving Averaged!')
            stored_loss = val_loss2
            if epoch >= args.start_decay_at:
                print('Reducing learning rate')
                lr = lr * args.decay_multiplier
        else:
            print('Reducing learning rate')
            lr = lr * args.decay_multiplier

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
