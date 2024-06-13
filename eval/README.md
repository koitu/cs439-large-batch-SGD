default parameters:
- architecture: resnet18
- lr: 0.1 * (mini-batch size) / 256
- momentum: 0.9
- weight_decay: 5e-4
- warm up epochs: 5
- workers: 4

- s1: train for 150, divide lr by 2.5 every 30, 60, 90, 110, 130
  - s1_nesterov: train with nesterov momentum
  - s1_no_warmup
- s2: train for 200, divide lr by 5 at 60, 120, 160
- s3: train for 300, divide lr by 10 at 150, 225
- s4: train for 200, and linearly interpolate to 0
- s4_alt: train for 100, and linearly interpolate to 0

only the times for s1_b*, s1_b*_no_warmup, s1_b*_nesterov, and s2_b* are accurate

the rest were either run on different machines or had two models training at the same time
