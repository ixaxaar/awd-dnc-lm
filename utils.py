from torch.autograd import Variable

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    elif type(h) == tuple:
        return tuple(repackage_hidden(v) for v in h)
    elif type(h) == list:
        return [ repackage_hidden(v) for v in h ]
    else:
      return h

def repackage_hidden_dnc(h):
  if h is None:
    return None

  if type(h) is list and h[0] is None:
    return [None]

  (chx, mhxs, _) = h[0]
  chx = repackage_hidden(chx)
  if type(mhxs) is list:
    mhxs = [dict([(k, repackage_hidden(v)) for k, v in mhx.items()]) for mhx in mhxs]
  else:
    mhxs = dict([(k, repackage_hidden(v)) for k, v in mhxs.items()])
  return [(chx, mhxs, None)]


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda != -1:
        data = data.cuda(args.cuda)
    return data

def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target
