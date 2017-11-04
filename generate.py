###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable
import numpy as np

import data
from visdom import Visdom

viz = Visdom()

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.cuda == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda <GPU_ID>")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda != -1:
    model.cuda(args.cuda)
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda(args.cuda)

with open(args.outf, 'w') as outf:
    debug = []

    mem_debug = {
        "memory": [],
        "link_matrix": [],
        "precedence": [],
        "read_weights": [],
        "write_weights": [],
        "usage_vector": [],
    }


    for i in range(args.words):
        output, hidden, v = model(input, hidden, reset_experience=False)
        debug.append(v)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))


    mem_debug["memory"] += [ x[0]["memory"] for x in debug ]
    mem_debug["link_matrix"] += [ x[0]["link_matrix"] for x in debug ]
    mem_debug["precedence"] += [ x[0]["precedence"] for x in debug ]
    mem_debug["read_weights"] += [ x[0]["read_weights"] for x in debug ]
    mem_debug["write_weights"] += [ x[0]["write_weights"] for x in debug ]
    mem_debug["usage_vector"] += [ x[0]["usage_vector"] for x in debug ]

    print("=====================================================")
    print(model)
    print("=====================================================")
    mem_debug = { k: np.array(v) for k,v in mem_debug.items() }
    mem_debug = { k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k,v in mem_debug.items() }

    viz.heatmap(
        mem_debug['memory'],
        opts=dict(
            xtickstep=10,
            ytickstep=2,
            title='Memory, t: ' + str(i),
            ylabel='layer * time',
            xlabel='mem_slot * mem_size'
        )
    )

    viz.heatmap(
        mem_debug['link_matrix'],
        opts=dict(
            xtickstep=10,
            ytickstep=2,
            title='Link Matrix, t: ' + str(i),
            ylabel='layer * time',
            xlabel='mem_slot * mem_slot'
        )
    )

    viz.heatmap(
        mem_debug['precedence'],
        opts=dict(
            xtickstep=10,
            ytickstep=2,
            title='Precedence, t: ' + str(i),
            ylabel='layer * time',
            xlabel='mem_slot'
        )
    )

    viz.heatmap(
        mem_debug['read_weights'],
        opts=dict(
            xtickstep=10,
            ytickstep=2,
            title='Read Weights, t: ' + str(i),
            ylabel='layer * time',
            xlabel='nr_read_heads * mem_slot'
        )
    )

    viz.heatmap(
        mem_debug['write_weights'],
        opts=dict(
            xtickstep=10,
            ytickstep=2,
            title='Write Weights, t: ' + str(i),
            ylabel='layer * time',
            xlabel='mem_slot'
        )
    )

    viz.heatmap(
        mem_debug['usage_vector'],
        opts=dict(
            xtickstep=10,
            ytickstep=2,
            title='Usage Vector, t: ' + str(i),
            ylabel='layer * time',
            xlabel='mem_slot'
        )
    )

