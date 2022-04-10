# Few-shot Segmentation Experiments

Adopted from [**ASGNet**](https://github.com/Reagan1311/ASGNet).

---

## Requisites
- Test Env: Python 3.9.7 (Singularity)
- Packages:
    - torch (1.10.2+cu113), torchvision (0.11.3+cu113)
    - numpy, scipy, pandas, tensorboardX
    - cv2

---

## Clone codebase
```
cd /scratch/$USER
git clone https://github.com/TeamOfProfGuo/FSS-Exp
cd FSS-Exp
```

---

## Preparation

### Pascal-5i dataset
**Note:** Make sure the path in prepare_dataset.sh works for you.
```
cd /scratch/$USER/FSS-Exp
bash prepare_dataset.sh
```

### Pretrained models
Download via <a href="https://drive.google.com/file/d/1rMPedZBKFXiWwRX3OHttvKuD1h9QRDbU/view?usp=sharing" target="_blank">this link</a>, and transfer the zip file to your project root on Greene.
```
cd /scratch/$USER/FSS-Exp
unzip initmodel.zip
```

### Config file
Modify the **data_root** under *config/pascal/pascal_asgnet_split0_resnet50.yaml*.

---

## Training
**Note:** Modify the path in slurm scripts (as needed) before you start.
```
# switch to project root
cd /scratch/$USER/FSS-Exp

# train & save & test
sbatch train.slurm pascal asgnet split0_resnet50

# After the job ends:
# [To be updated]
```