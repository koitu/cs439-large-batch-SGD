#!/bin/bash

python main.py --name b0032 -b 32
python main.py --name b0128 -b 128
python main.py --name b0256 -b 256
python main.py --name b0512 -b 512
python main.py --name b1024 -b 1024
python main.py --name b2048 -b 2048
python main.py --name b4096 -b 4096
python main.py --name b8192 -b 8192

python main.py --name b0032_no_warmup -w 0 -b 32
python main.py --name b0128_no_warmup -w 0 -b 128
python main.py --name b0256_no_warmup -w 0 -b 256
python main.py --name b0512_no_warmup -w 0 -b 512
python main.py --name b1024_no_warmup -w 0 -b 1024
python main.py --name b2048_no_warmup -w 0 -b 2048
python main.py --name b4096_no_warmup -w 0 -b 4096
python main.py --name b8192_no_warmup -w 0 -b 8192
