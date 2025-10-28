### First batch of experiments (epochs=5)
| Experiment name | Train accuracy | Time (sec/it) | Total training time (approx.)| 
|:---------------:|:--------------:|:-------------:|:-------------------:|
| Alternating horizontal flips | 33.21% | 29 | 2min 41s | 
| Removed alt horizontal flips | 46.71% | 19.54 | 2min 2s |
| Random horizontal flips | 53.21% | 22.29 | 2min 26s | 

### First batch of experiments (variable epochs)
| Experiment name | Number of epochs | Train accuracy | Time (sec/it) | Total training time (approx.)| 
|:---------------:|:----------------:|:--------------:|:-------------:|:-------------------:|
| Random horizontal flips | 20 | 58.61% | 18 | 6min | 
| + Random cutmix/mixup/identity | 50 | 68.09% | 17.40 | 14min 5s | 
| + Alternative flips | 50 | 79.71% | 22 | 18min 21s |

### Optimizer = AdamW(learning_rate=0.001, betas=(0.9,0.999), weight-decay=0.01) (default)
| Transforms | Number of epochs | Train accuracy | Validation accuracy | Total training time (approx.)| 
|:----------:|:----------------:|:--------------:|:-------------------:|:----------------------------:|
| padding, crop, random horizontal flip | 20 | 85.68% | 17.70% | 7min 10s |
| no runtime transforms | 20 | 98.30% | 69.50% | 3min 17s |

learning rate >= 0.1 merge teribil (max train_acc ~3%)
## Experiments for lr
- Train: 83.62, Validation: 40.7200, Best: 83.62 at epoch 2 for lr=0.01: 3/3 [00:35<00:00, 11.70s/it]
- Train: 99.35, Validation: 46.8000, Best: 99.35 at epoch 2 for lr=0.001: 3/3 [00:35<00:00, 11.69s/it]
- Train: 99.83, Validation: 47.0600, Best: 99.83 at epoch 2 for lr=0.0001: 3/3 [00:35<00:00, 11.67s/it]
## Experiments for wd
- Train: 99.87, Validation: 46.9100, Best: 99.87 at epoch 2 for wd=0.5: 3/3 [00:35<00:00, 11.83s/it]
- Train: 99.93, Validation: 46.9000, Best: 99.93 at epoch 2 for wd=0.005: 3/3 [00:35<00:00, 11.76s/it]
- Train: 99.94, Validation: 46.8300, Best: 99.95 at epoch 0 for wd=0.0005: 3/3 [00:35<00:00, 11.84s/it]
## Experiments for betas
- Train: 99.97, Validation: 46.80, Best (val): 46.81 at epoch 0 for betas=(0.9, 0.999): 3/3 [00:35<00:00, 11.84s/it]
- Train: 99.97, Validation: 46.86, Best (val): 46.86 at epoch 2 for betas=(0.8, 0.888): 3/3 [00:35<00:00, 11.86s/it]
- Train: 99.98, Validation: 46.67, Best (val): 46.79 at epoch 0 for betas=(0.7, 0.777): 3/3 [00:35<00:00, 11.85s/it]
- Train: 99.98, Validation: 46.53, Best (val): 47.04 at epoch 1 for betas=(0.5, 0.999): 3/3 [00:35<00:00, 11.91s/it]
- Train: 1.63, Validation: 1.51, Best (val): 42.89 at epoch 0 for betas=(0.999, 0.5): 3/3 [00:35<00:00, 11.70s/it]  
0.0
## Experiments for label smoothing
### With runtime transforms
Train: 51.21, Validation: 11.08, Best: 12.32 at epoch 3: 5/5 [01:52<00:00, 22.41s/it] for smoothing=0.1
Train: 64.44, Validation: 10.24, Best: 16.50 at epoch 0: 5/5 [01:51<00:00, 22.34s/it] for smoothing=0.3
Train: 73.57, Validation: 18.43, Best: 18.59 at epoch 1: 5/5 [01:52<00:00, 22.46s/it] for smoothing=0.6
Train: 80.59, Validation: 17.75, Best: 20.78 at epoch 2: 5/5 [01:52<00:00, 22.57s/it] for smoothing=0.8
### Without
Train: 99.75, Validation: 70.43, Best: 73.56 at epoch 3: 5/5 [00:58<00:00, 11.62s/it] for smoothing=0.1
Train: 99.73, Validation: 70.18, Best: 72.43 at epoch 0: 5/5 [00:58<00:00, 11.64s/it] for smoothing=0.3
Train: 99.90, Validation: 70.17, Best: 73.47 at epoch 2: 5/5 [00:58<00:00, 11.63s/it] for smoothing=0.6
Train: 99.96, Validation: 69.57, Best: 73.40 at epoch 0: 5/5 [00:58<00:00, 11.73s/it] for smoothing=0.8

Best Train: 98.69, Best Validation: 52.37 at epoch 1: 5/5 [00:58<00:00, 11.63s/it]
Best Train: 99.75, Best Validation: 56.66 at epoch 0: 5/5 [00:58<00:00, 11.64s/it]
Best Train: 99.95, Best Validation: 58.77 at epoch 0: 5/5 [00:58<00:00, 11.68s/it]
Best Train: 99.96, Best Validation: 59.48 at epoch 0: 5/5 [00:58<00:00, 11.63s/it]

Best Train: 87.15, Best Validation: 86.17 at epoch 0:  20%|██        | 1/5 [00:15<01:00, 15.04s/it]dude how???

### Experiments over 60%
| Experiment name | Transforms | Number of epochs | Train accuracy | Validation accuracy | Total training time (approx.)| 
|:---------------:|:----------:|:----------------:|:--------------:|:-------------------:|:----------------------------:|
|StepLR + cutmix-mixup| none | 25 | 83.92% | 60.84% | 8m 31s |
| += label_smoothing=0.3 | none | 25 | 83.55% | 60.62% | 8m 29s | 
| -= label_smoothing | RandomHorizontalFlip + RandomRotation(10) | 25 | 70.90% | 64.33% | 19m 50s |
| -= label_smoothing | -rotation, +padding+crop | 25 | 62.80% | 65.85% | 14m 5s |




Conclusions:
-> use augmentation methods 

