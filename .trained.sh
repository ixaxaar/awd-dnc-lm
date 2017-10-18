#!/usr/bin/env bash


# fate - 400-37.19-34.70
python main.py --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB.pt

# 512x2-30.89-28.29
python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512.pt --log-interval 20

python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 50 --data data/wikitext-2 --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --log-interval 20 --save WT2-512.pt


python ./generate.py --data data/penn --checkpoint PTB-512x2-30.89-28.29.pt --words 1000 --cuda --outf generated.txt