# Boosting-Crowd-Counting-via-Multifaceted-Attention
Official Implement of CVPR 2022 paper 'Boosting Crowd Counting via Multifaceted Attention'

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train
1. Dowload Dataset JHU++ or UCF-QNRF.
2. Preprocess them by 'preprocess_dataset.py' or 'preprocess_dataset_ucf.py'.
3. Change the path to where your data and models are located in 'Train.py'.
4. Run 'Train.py'
5. Wait patiently for the program to finish.
6. Then you will get a good counting model!

## Two-stage Swin-Large training (clean → hazy)

Prepare dataset roots with matching `.jpg` images and `.npy` annotations in
the format expected by `datasets/crowd.py`:

```text
sha/
├── clean/train/
├── clean/val/
├── hazy/train/
└── mix/val/
```

The baseline stage updates the entire Swin-Large backbone and the original
counting backend on `clean`. It requires the official
`swin_large_patch4_window12_384_22k.pth` checkpoint. The LoRA stage loads the
*entire* baseline model, freezes the Swin backbone, and trains LoRA plus the
counting backend on `hazy/train`, validating on `mix/val`. Both stages keep the
Bayesian density-map loss and use 384×384 crops.

```bash
python train.py --stage baseline --seed 42 \
  --data-dir "/media/mmslab5090/SSD2/crowd counting test/sha/clean" \
  --swin-pretrained /path/to/swin_large_patch4_window12_384_22k.pth \
  --save-dir model --batch-size 1

python train.py --stage lora --seed 42 \
  --train-dir "/media/mmslab5090/SSD2/crowd counting test/sha/hazy" \
  --val-dir "/media/mmslab5090/SSD2/crowd counting test/sha/mix" \
  --baseline-checkpoint /path/to/baseline/best_model.pth \
  --save-dir model --batch-size 1
```

The three dataset paths above are CLI defaults. In the LoRA stage, passing
`--train-dir ''` or `--val-dir ''` selects `hazy/` or `mix/` beside `data-dir`.

Runs are saved separately under `model/baseline/` and `model/lora/`. You may
use a baseline epoch `.tar` checkpoint instead of `best_model.pth`. `--resume`
is only for continuing an interrupted run *within the same stage*; it does
not convert a baseline run into a LoRA run. Validation begins at epoch 0 by
default, so `best_model.pth` can be produced during the baseline stage.
The seed defaults to 42; use the same `--seed` value when resuming a stage.
Training and validation show per-epoch `tqdm` progress bars.


## Test
1. Dowload Dataset JHU++ or UCF-QNRF.
2. Preprocess them by 'preprocess_dataset.py' or 'preprocess_dataset_ucf.py'.
3. JHU Model [Link](https://drive.google.com/file/d/14piGsWRFy9BSXI1Jv9zRxypDxpOHbwCY/view?usp=sharing); UCF Model [Link](https://drive.google.com/file/d/1Y2WU0kIlZq3x28JZskvGx1cuZt0KQkXF/view?usp=sharing)
4. Change the path to where your data and models are located in 'Test.py'.
5. Run 'Test.py'.


## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{lin2022boosting,
  title={Boosting Crowd Counting via Multifaceted Attention},
  author={Lin, Hui and Ma, Zhiheng and Ji, Rongrong and Wang, Yaowei and Hong, Xiaopeng},
  booktitle={CVPR},
  year={2022}
}
```
