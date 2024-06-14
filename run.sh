#!/bin/bash

# s1
python main.py --name s1_b0064 -s 1 -b 64
python main.py --name s1_b0128 -s 1 -b 128
python main.py --name s1_b0256 -s 1 -b 256
python main.py --name s1_b0512 -s 1 -b 512
python main.py --name s1_b1024 -s 1 -b 1024
python main.py --name s1_b2048 -s 1 -b 2048
python main.py --name s1_b4096 -s 1 -b 4096
python main.py --name s1_b8192 -s 1 -b 8192

# s1 without warmup
python main.py --name s1_b0064_no_warmup -s 1 -w 0 -b 64
python main.py --name s1_b0128_no_warmup -s 1 -w 0 -b 128
python main.py --name s1_b0256_no_warmup -s 1 -w 0 -b 256
python main.py --name s1_b0512_no_warmup -s 1 -w 0 -b 512
python main.py --name s1_b1024_no_warmup -s 1 -w 0 -b 1024
python main.py --name s1_b2048_no_warmup -s 1 -w 0 -b 2048
python main.py --name s1_b4096_no_warmup -s 1 -w 0 -b 4096
python main.py --name s1_b8192_no_warmup -s 1 -w 0 -b 8192

# s2
python main.py --name s2_b0064 -s 2 -b 64
python main.py --name s2_b0128 -s 2 -b 128
python main.py --name s2_b0256 -s 2 -b 256
python main.py --name s2_b0512 -s 2 -b 512
python main.py --name s2_b1024 -s 2 -b 1024
python main.py --name s2_b2048 -s 2 -b 2048
python main.py --name s2_b4096 -s 2 -b 4096
python main.py --name s2_b8192 -s 2 -b 8192

# s3
python main.py --name s3_b0064 -s 3 -b 64
python main.py --name s3_b0128 -s 3 -b 128
python main.py --name s3_b0256 -s 3 -b 256
python main.py --name s3_b0512 -s 3 -b 512
python main.py --name s3_b1024 -s 3 -b 1024
python main.py --name s3_b2048 -s 3 -b 2048
python main.py --name s3_b4096 -s 3 -b 4096
python main.py --name s3_b8192 -s 3 -b 8192

# s4
python main.py --name s4_b0064 -s 4 -b 64
python main.py --name s4_b0128 -s 4 -b 128
python main.py --name s4_b0256 -s 4 -b 256
python main.py --name s4_b0512 -s 4 -b 512
python main.py --name s4_b1024 -s 4 -b 1024
python main.py --name s4_b2048 -s 4 -b 2048
python main.py --name s4_b4096 -s 4 -b 4096
python main.py --name s4_b8192 -s 4 -b 8192

# s4 without warmup
python main.py --name s4_b0064 -s 4 -w 0 -b 64
python main.py --name s4_b0128 -s 4 -w 0 -b 128
python main.py --name s4_b0256 -s 4 -w 0 -b 256
python main.py --name s4_b0512 -s 4 -w 0 -b 512
python main.py --name s4_b1024 -s 4 -w 0 -b 1024
python main.py --name s4_b2048 -s 4 -w 0 -b 2048
python main.py --name s4_b4096 -s 4 -w 0 -b 4096
python main.py --name s4_b8192 -s 4 -w 0 -b 8192

# s4 alt with warmup
python main.py --name s4_b0064 -s 5 -b 64
python main.py --name s4_b0128 -s 5 -b 128
python main.py --name s4_b0256 -s 5 -b 256
python main.py --name s4_b0512 -s 5 -b 512
python main.py --name s4_b1024 -s 5 -b 1024
python main.py --name s4_b2048 -s 5 -b 2048
python main.py --name s4_b4096 -s 5 -b 4096
python main.py --name s4_b8192 -s 5 -b 8192
