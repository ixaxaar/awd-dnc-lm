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
from visdom import Visdom

import data

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
parser.add_argument('--input', type=str, default='./data/penn/test.txt',
                    help='Input file')

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

matches = []

with open(args.input, 'r') as f:
  for raw_input in f:
    try:
      raw_input = raw_input.split()
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
      out_words = []

      if model.debug:
          output, hidden, v = model(input, hidden, reset_experience=True)
          debug.append(v)
      else:
          output, hidden = model(input, hidden, reset_experience=True)

      word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

      word_idx = [ torch.topk(w, args.nbest)[1] for w in word_weights ]
      words = [ [ corpus.dictionary.idx2word[w] for w in ws ] for ws in word_idx ]

      total_len = len(words)
      matched = 0
      for i,w in enumerate(words):
          if raw_input[i+1] in w:
              matched += 1
    except Exception as e:
      pass

    matches.append(matched / (total_len))
    print("Matched: %s / %s" % (matched, total_len))

  viz.histogram(
      X=np.array(matches),
      opts=dict(
          colormap='Viridis',
          title='Fraction of Words Predicted.',
          ylabel='Number of Sentences',
          xlabel='Fraction Matched',
          numbins=30
      )
  )
