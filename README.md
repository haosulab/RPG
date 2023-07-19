# Reparameterized Policy Learning

### [Project Page](https://haosulab.github.io/RPG/) | [Paper](icml2023_rpg_camera_ready.pdf)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/RPG/blob/main/tiny_rpg/tiny-rpg-bandit.ipynb)

This is the official implementation in PyTorch of paper:

### Reparameterized Policy Learning for Multimodal Trajectory Optimization
Zhiao Huang, Litian Liang, Zhan Ling, Xuanlin Li, Chuang Gan, Hao Su

ICML 2023 (Oral Presentation)

![](tiny_rpg/tiny_rpg.gif)


use the below command for running sparse and dense reward experiments
```
cd run
python3 mbrpg.py --env_name EEArm --exp rpgcv2 --seed 0
python3 mbrpg.py --env_name AntPushDense --exp dense --seed 0
```

