###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file is used to visualize memory contents in visdom
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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nbest', type=int, default=5,
                    help='top n results')
parser.add_argument('--cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--debug', action='store_true',
                    help='debug DNC memory contents in visdom (on localhost)')
parser.add_argument('--input', type=str, default='shares of ual the parent of united airlines were extremely active all day friday reacting to news and rumors about the proposed $ N billion buy-out of the airline by an <unk> group',
                    help='Input sentence to debug')

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

raw_input = args.input.split()
input = Variable(torch.Tensor([ corpus.dictionary.word2idx[x] for x in raw_input ]).long(), volatile=True).unsqueeze(1)
raw_input.append('<eos>')

# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda != -1:
    input = input.cuda(args.cuda)

mem_debug = {
    "memory": [],
    "link_matrix": [],
    "precedence": [],
    "read_weights": [],
    "write_weights": [],
    "usage_vector": [],
}

debug = []

if model.debug:
    output, hidden, v = model(input, hidden, reset_experience=True)
    debug.append(v)
else:
    output, hidden = model(input, hidden, reset_experience=True)

word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
ppx = [ torch.topk(w, args.nbest)[0].numpy() for w in word_weights ]
ppx = np.array(ppx).tolist()
ppx = [ [ pp / max(p) for pp in p ] for p in ppx]

word_idx = [ torch.topk(w, args.nbest)[1] for w in word_weights ]
words = [ [ corpus.dictionary.idx2word[w] for w in ws ] for ws in word_idx ]

mem_debug = v[0]
total_len = len(words)
matched = 0
# layer1_inputs = []
# layern_outputs = []
# for i,w in enumerate(words):
#     layer1_inputs.append(str(i)+"             "+raw_input[i+1])
#     if raw_input[i+1] in w:
#         layern_outputs.append(str(i+1)+"**"+"             "+raw_input[i+1])
#         matched += 1
#     else:
#         layern_outputs.append(str(i+1)+"**"+"             "+"("+w[0]+")")

out_words = layer1_inputs + layern_outputs
print("Matched: %s / %s" % (matched, total_len-1))

if args.debug:

    print("=====================================================")
    print(model)
    print("=====================================================")


    # interlace layers
    # n = model.nlayers
    # mem_debug = { k: v.reshape((n, int(v.shape[0]/n), v.shape[1])).transpose((1,0,2)).reshape((v.shape[0], v.shape[1])) for k,v in mem_debug.items() }

    for k,v in mem_debug.items():
        print(k, v.shape)

    viz.contour(
        X=ppx,
        opts=dict(
            colormap='Viridis',
            title='DNC output',
            ylabel='Time',
            xlabel='Top k values',
            markers=True,
            fillarea=False
        )
    )

    viz.heatmap(
        mem_debug['memory'],
        opts=dict(
            colormap='Viridis',
            title='Memory',
            ylabel='Controller Layer * Time',
            xlabel='Number of memory cells * Size of Memory',
            rownames=out_words,
            xmin=0,
            xmax=0.5
        )
    )

    viz.heatmap(
        mem_debug['link_matrix'],
        opts=dict(
            colormap='Viridis',
            rownames=out_words,
            xmin=0,
            xmax=0.5,
            title='Link Matrix',
            ylabel='Controller Layer * Time',
            xlabel='Number of memory cells * Number of memory cells'
        )
    )

    viz.heatmap(
        mem_debug['precedence'],
        opts=dict(
            colormap='Viridis',
            rownames=out_words,
            xmin=0,
            xmax=0.5,
            title='Precedence',
            ylabel='Controller Layer * Time',
            xlabel='Number of memory cells'
        )
    )

    viz.heatmap(
        mem_debug['read_weights'],
        opts=dict(
            colormap='Viridis',
            rownames=out_words,
            xmin=0,
            xmax=0.5,
            title='Read Weights',
            ylabel='Controller Layer * Time',
            xlabel='Number of Read Heads * Number of memory cells'
        )
    )

    viz.heatmap(
        mem_debug['write_weights'],
        opts=dict(
            rownames=out_words,
            xmin=0,
            xmax=0.5,
            colormap='Viridis',
            title='Write Weights',
            ylabel='Controller Layer * Time',
            xlabel='Number of memory cells'
        )
    )

    viz.heatmap(
        mem_debug['usage_vector'],
        opts=dict(
            colormap='Viridis',
            rownames=out_words,
            xmin=0,
            xmax=0.5,
            title='Usage Vector',
            ylabel='Controller Layer * Time',
            xlabel='Number of memory cells'
        )
    )

