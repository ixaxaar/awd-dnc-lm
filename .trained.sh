#!/usr/bin/env bash


# # fate - 400-37.19-34.70
# python main.py --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB.pt

# # 512x2-30.89-28.29
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512.pt --log-interval 20

# python main.py --nhid 512 --nlayers 10 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 10 --batch_size 30 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512x10.pt --log-interval 20

# # no reset experience 512x2-49.92-
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512.pt --log-interval 20

# # LSTM-512x2-85.31-79.95
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model LSTM --epoch 50 --cuda 0 --save PTB-LSTM-512x2.pt --log-interval 20


# # with lr annealing - does not work
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512.pt --log-interval 20

# # 512-2-25.69-24.45
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 50 --data data/wikitext-2 --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --log-interval 20 --save WT2-512.pt


# python ./generate.py --data data/penn --checkpoint PTB-512x2-30.89-28.29.pt --words 1000 --cuda --outf generated.txt

# python ./generate.py --data data/wikitext-2 --checkpoint WT2-512-2-25.69-24.45.pt --words 1000 --cuda --outf generated-wikitext-2.txt

# # DNC v0.0.4 512-2-20.73-19.17
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 10 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-bptt-500.pt --log-interval 20 --bptt 500

# # DNC v0.0.5 512-2x2-38.05-34.99
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 4 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-dncv0.0.5-cuda1.pt --log-interval 20

# # num hidden layers = 1 512-2x1-45.44-42.28
# python main.py --nhid 512 --nlayers 2 --emsize 512 --cell_size 512 --read_heads 2 --nr_cells 8 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-dncv0.0.5-cuda1.pt --log-interval 20

# python main.py --nhid 512 --nlayers 4 --emsize 512 --cell_size 512 --read_heads 2 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-4-dncv0.0.5-cuda0.pt --log-interval 20

# python main.py --nhid 512 --nlayers 4 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --log-interval 20


# # visdom discussion - 52.29-48.72
# python main.py --nhid 512 --debug --nlayers 2 --nhlayers 2 --emsize 512 --cell_size 512 --read_heads 4 --nr_cells 8 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --log-interval 20

# python ./generate.py --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --words 20 --cuda 1 --outf generated.txt

# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --cuda 1 --nbest 10 --input "the american national"

# # 11-gram match test
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --cuda 1 --nbest 10 --input "for the nine-month period intel reported net of $ N million or $ N a share down N N from $ N million or $ N a share"

# # resistance to stay on a manifold:
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --cuda 1 --nbest 10 --input "for the nine-month period intel reported net of $ N million or $ N a federal down N N from $ N million or $ N a share"
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --cuda 1 --nbest 10 --input "for the nine-month period intel reported net of $ N million or $ N a should sales continue to be strong N N from $ N million or $ N a share"
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt --cuda 1 --nbest 10 --input "u.s. holders now own more than N N of blue arrow compared with N N last january"
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-4-dncv0.0.5-cuda1-debug.pt  --cuda 1 --nbest 10 --input "el salvador 's government opened a new round of talks with the country 's leftist rebels in an effort to end a <unk> civil war"

# # loweest ppx
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "the american national"

# # trained 11-gram
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "for the nine-month period intel reported net of $ N million or $ N a share down N from $ N million or $ N a share"

# # 11-gram match test, extra N
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "for the nine-month period intel reported net of $ N million or $ N a share down N N from $ N million or $ N a share"

# # resistance to stay on a manifold:
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "for the nine-month period intel reported net of $ N million or $ N a federal down N N from $ N million or $ N a share"
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "for the nine-month period intel reported net of $ N million or $ N a should sales continue to be strong N N from $ N million or $ N a share"

# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "$ N million or $ N a share down from $ N million or $ N a share"
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "subscribers not ad watches meet as or N telephones increasing rates who advertisers"

# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "$ N million or $ N a share down from $ N telephones increasing rates who advertisers million or $ N a share"

# # N billion from $ N billion, president and chief executive officer of
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "N billion from $ N billion president and chief executive officer of"
# # $ N million or N cents, in new york stock exchange composite, N N to yield N N, president and chief executive officer of
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "$ N million or N cents in new york stock exchange composite N N to yield N N president and chief executive officer of"

# python topk_matches.py --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input ./data/penn/test.txt

# # ANAPHORA
# # python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --cuda 0 --nbest 10 --input "but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged N points most of it in the final hour it barely managed to stay this side of chaos"


# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --words 100 --cuda 0 --outf generated.txt --input "u.s. holders now own more than N N of blue arrow compared with N N last january"
# python ./debug.py --debug --seed 121 --data data/penn --checkpoint PTB-512-layers-2x2-cells_size-64-nr_cells-16-cuda0.pt --words 100 --cuda 0 --outf generated.txt --input "el salvador 's government opened a new round of talks with the country 's leftist rebels in an effort to end a <unk> civil war"



# # layers=4 not done
# # controller layers
# # stopped at i = 6
# for (( i = 2; i < 10; i++ )); do
#   python main.py --nhid 512 --nlayers ${i} --nhlayers 2 --emsize 512 --cell_size 256 --read_heads 4 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-${i}-dncv0.0.5-cuda1.pt --log-interval 20 > PTB-512-layers-${i}-dncv0.0.5-cuda1.log
# done

# # 1,2,3 done
# # controller layer hidden layers
# for (( i = 4; i < 5; i++ )); do
#   python main.py --nhid 512 --nlayers 2 --nhlayers ${i} --emsize 512 --cell_size 256 --read_heads 4 --nr_cells 8 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x${i}-dncv0.0.5-cuda0.pt --log-interval 20 > PTB-512-layers-2x${i}-dncv0.0.5-cuda0.log
# done

# # stopped at 14
# # memory cells
# for (( i = 2; i < 20; i++ )); do
#   python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --cell_size 256 --read_heads 4 --nr_cells $((i*2)) --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x2-cells-$((i*2))-cuda0.pt --log-interval 20 > PTB-512-layers-2x2-cells-$((i*2))-cuda0.log
# done

# i=1
# python main.py --nhid 512 --nlayers 2 --nhlayers 1 --emsize 512 --cell_size 256 --read_heads 4 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x1-dncv0.0.5-cuda0.pt --log-interval 20 > PTB-512-layers-2x1-dncv0.0.5-cuda0.log

# i=4
# python main.py --nhid 512 --nlayers ${i} --nhlayers 2 --emsize 512 --cell_size 256 --read_heads 4 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-${i}-dncv0.0.5-cuda1.pt --log-interval 20 > PTB-512-layers-${i}-dncv0.0.5-cuda1.log
# i=1
# python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --cell_size 256 --read_heads 4 --nr_cells $((i*2)) --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-2x2-cells-$((i*2))-cuda1.pt --log-interval 20 > PTB-512-layers-2x2-cells-$((i*2))-cuda1.log

# # memory cell size vs nr cells
# for (( i = 1; i < 7; i++ )); do
#   python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --debug --cell_size $(( 512 / 2**$i )) --read_heads 4 --nr_cells $((2**$i*2)) --cell_size $(( 512 / 2**$i )) --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-nr_cells-$((2**$i*2))-cuda0.pt --log-interval 20 > PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-nr_cells-$((2**$i*2))-cuda0.log
# done

# # another memory scaling
# for (( i = 2; i < 7; i++ )); do
#   python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --debug --cell_size $(( 512 / 2**$i )) --read_heads 4 --nr_cells $((32*$i)) --cell_size $(( 512 / 2**$i )) --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-nr_cells-$((32*$i))-cuda1.pt --log-interval 20 > PTB-512-layers-2x2-cells_size-$(( 512 / 2**$i ))-nr_cells-$((32*$i))-cuda1.log
# done

# # SOTA
# python main.py --nhid 512 --nlayers 3 --nhlayers 1 --emsize 512 --cell_size 64 --read_heads 4 --nr_cells 16 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-cuda1.pt --log-interval 20 > PTB-SOTA-cuda1.log
# python main.py --nhid 512 --nlayers 3 --nhlayers 2 --emsize 512 --cell_size 64 --read_heads 4 --nr_cells 16 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA2-cuda1.pt --log-interval 20 > PTB-SOTA2-cuda1.log

# python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --cell_size 64 --read_heads 4 --nr_cells 16 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA3-cuda1.pt --log-interval 20 > PTB-SOTA3-cuda1.log

# # Actual SOTA decider
# python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --debug --cell_size 64 --read_heads 4 --nr_cells 16 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-decider-cuda1.pt --log-interval 20 > PTB-SOTA-decider-debug-cuda1.log

# python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --debug --cell_size 512 --read_heads 4 --nr_cells 8 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 0 --save PTB-SOTA-decider1-cuda0.pt --log-interval 20 > PTB-SOTA-decider1-debug-cuda0.log

# # linzen
# python main.py --nhid 512 --nlayers 3 --nhlayers 2 --emsize 512 --cell_size 64 --read_heads 4 --nr_cells 16 --batch_size 50 --data data/linzen --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save Linzen-SOTA2-cuda1.pt --log-interval 20 > Linzen-SOTA2-cuda1.log

# python main.py --nhid 512 --nlayers 3 --nhlayers 1 --emsize 512 --cell_size 64 --read_heads 4 --nr_cells 16 --batch_size 50 --data data/wikitext-2 --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save WT2-SOTA-cuda1.pt --log-interval 20 > WT2-SOTA-cuda1.log

# bptt






## DHOKAAAA!

# baseline 82.96 / 78.77
python main.py --nhid 1024 --nlayers 2 --nhlayers 1 --emsize 400 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model LSTM --epoch 50 --cuda 1 --save PTB-LSTM-decider1-cuda1.pt --log-interval 20

# DNC without memory
python main.py --nhid 1024 --nlayers 2 --nhlayers 1 --emsize 400 --debug --cell_size 1 --read_heads 1 --nr_cells 1 --batch_size 100 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-decider1-cuda1.pt --log-interval 20

# 92.54 / 90.28
python main.py --nhid 1024 --nlayers 1 --nhlayers 2 --emsize 400 --debug --cell_size 64 --read_heads 4 --nr_cells 16 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-decider-cuda1.pt --log-interval 20

# python main.py --nhid 512 --nlayers 2 --nhlayers 2 --emsize 512 --debug --cell_size 512 --read_heads 4 --nr_cells 8 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-decider-cuda1.pt --log-interval 20


# 256*16 92.89
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 256 --read_heads 4 --nr_cells 16 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-decider-cuda1.pt --log-interval 20

#  wdrop 94.68 / 89.12
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 512 --read_heads 4 --nr_cells 8 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-decider-cuda1.pt --log-interval 20

# 105.39 / 99.85 / 13
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 128 --read_heads 8 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-read_heads-16-nr_cells-64-cuda1.pt --log-interval 20

###########################
# new dnc
###########################
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 128 --read_heads 8 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-read_heads-16-nr_cells-64-cuda1.pt --log-interval 20


python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 50 --read_heads 2 --nr_cells 200 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-read_heads-2--cell_size-50-nr_cells-200-cuda1.pt --log-interval 20

# 105.77 / 99.85 /  12
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 128 --read_heads 16 --nr_cells 32 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-read_heads-16-nr_cells-64-cuda1.pt --log-interval 20

# 93.43 / 88.28 / 24
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 128 --read_heads 8 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-read_heads-16-nr_cells-64-cuda1.pt --log-interval 20

python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 256 --read_heads 4 --nr_cells 32 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-read_heads-4-nr_cells-32-cell_size-256-cuda1.pt --log-interval 20

# 94.36 / 88.50
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 128 --read_heads 8 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 50 --cuda 1 --save PTB-SOTA-read_heads-16-nr_cells-64-cuda1.pt --log-interval 20

# 100.03 / 95.47
python main.py --lr 0.0001 --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 50 --read_heads 4 --nr_cells 150 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 150 --cuda 1 --save PTB-SOTA-read_heads-16-nr_cells-64-lr-0.0001-cuda1.pt --log-interval 20

# rmsprop - 101.97 / 96.88 / 80
python main.py --lr 0.0001 --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 50 --read_heads 4 --nr_cells 150 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 150 --cuda 1 --save PTB-SOTA-cell_size-50-read_heads-4-nr_cells-64-lr-0.0001-rmsprop-cuda1.pt --log-interval 20

# baseline rmsprop - 87.27 / 82.05
python main.py --lr 0.0001 --nhid 512 --nlayers 3 --emsize 512 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model LSTM --epoch 150 --cuda 0 --save PTB-baseline-rmsprop-cuda0.pt --log-interval 20

# 16 read heads - 96.54 / 92.26 / 24
python main.py --lr 0.0001 --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 50 --read_heads 4 --nr_cells 150 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 150 --cuda 1 --save PTB-SOTA-cell_size-50-read_heads-4-nr_cells-64-lr-0.0001-rmsprop-cuda1.pt --log-interval 20

# cell size 1024 - 96.99 / 91.46
python main.py --nhid 512 --nlayers 1 --nhlayers 3 --emsize 512 --debug --cell_size 1024 --read_heads 4 --nr_cells 20 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-read_heads-4-nr_cells-20-cell_size-1024-cuda1.pt --log-interval 20

# 2 layers - 99.67 / 91.91
python main.py --nhid 512 --nlayers 1 --nhlayers 2 --emsize 400 --debug --cell_size 512 --read_heads 4 --nr_cells 40 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-read_heads-4-nr_cells-40-cell_size-512-cuda1.pt --log-interval 20

# 2x1 layers 96.27 / 91.81
python main.py --nhid 512 --nlayers 2 --nhlayers 1 --emsize 512 --debug --cell_size 128 --read_heads 4 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-layers-2x1-read_heads-4-nr_cells-64-cuda1.pt --log-interval 20

# proportional 95.85 / 91.27
python main.py --nhid 500 --nlayers 1 --nhlayers 3 --emsize 300 --debug --cell_size 50 --read_heads 4 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-proportional.pt --log-interval 20

python main.py --nhid 2000 --nlayers 1 --nhlayers 3 --emsize 300 --debug --cell_size 50 --read_heads 4 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-proportional.pt --log-interval 20


#
python main.py --nhid 500 --optim rmsprop --lr 0.0001 --nlayers 1 --nhlayers 3 --emsize 300 --debug --cell_size 50 --read_heads 4 --nr_cells 100 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 250 --cuda 1 --save PTB-proportional.pt --log-interval 20

# start_decay_at 20
python main.py --nhid 500 --start-decay-at 20 --optim adadelta --lr 1 --nlayers 1 --nhlayers 3 --emsize 300 --cell_size 50 --read_heads 4 --nr_cells 100 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 250 --cuda 0 --save PTB-proportional.pt --log-interval 20 --seed 218

python main.py --nhid 500 --start-decay-at 20 --optim adam --lr 0.01 --nlayers 1 --nhlayers 3 --emsize 300 --cell_size 50 --read_heads 4 --nr_cells 100 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 250 --cuda 0 --save PTB-proportional.pt --log-interval 20 --seed 218


python main.py --nhid 1024 --start-decay-at 5 --optim adam --lr 0.01 --nlayers 1 --nhlayers 3 --emsize 300 --cell_size 50 --read_heads 4 --nr_cells 100 --batch_size 50 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 250 --cuda 0 --save PTB-proportional.pt --log-interval 20 --seed 218


# layers 2x2 - 98.38 / 94.79
python main.py --nhid 500 --nlayers 2 --nhlayers 2 --emsize 300 --debug --cell_size 50 --read_heads 4 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-proportional.pt --log-interval 20

# bigger proportional & bptt 96.56 / 92.38
python main.py --bptt 200 --nhid 700 --nlayers 1 --nhlayers 3 --emsize 500 --debug --cell_size 50 --read_heads 4 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-proportional.pt --log-interval 20

# RNNs
python main.py --nhid 500 --nlayers 1 --nhlayers 3 --emsize 300 --debug --cell_size 50 --read_heads 4 --nr_cells 100 --batch_size 64 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 250 --cuda 0 --save PTB-proportional.pt --log-interval 20

# try super big mem and super big bptt

# 2 read heads 96.90 / 91.91
python main.py --lr 0.0001 --nhid 500 --nlayers 1 --nhlayers 3 --emsize 300 --debug --cell_size 50 --read_heads 2 --nr_cells 256 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-proportional.pt --log-interval 20

#####################
# no reset experience
#####################

# 94.55 / 89.87
python main.py --nhid 500 --nlayers 1 --nhlayers 3 --emsize 300 --debug --cell_size 50 --read_heads 4 --nr_cells 64 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-proportional-no_reset_exp.pt --log-interval 20

# shitloads of memory - 95.60 / 91.20
python main.py --nhid 500 --nlayers 1 --nhlayers 3 --emsize 300 --debug --cell_size 50 --read_heads 4 --nr_cells 200 --batch_size 25 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --model DNC --epoch 25 --cuda 1 --save PTB-proportional-no_reset_exp.pt_cells_200.pt --log-interval 20








