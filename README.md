# Transverse Velocity Field Measurements in High-resolution Solar Images Based on Deep Learning(RAA, 2023)
The project is to calculate the Transverse velocity field of the solar image and achieve accurate optical flow estimation by calculating the pixel displacement in the image. 

Zhen-Hong Shang, Si-Yu Mu, Kai-Fan Ji and Zhen-Ping Qiang

Paper:(https://doi.org/10.1088/1674-4527/accbaf)

![image](https://github.com/lygmsy123/transverse-velocity-field-measurement/blob/main/PWCNet-H.png)
# This repository includes:
pretrained model and source code of our paper
# Requirements
The code has been tested with PyTorch 1.13 and Cuda 11.6.
```Shell
conda create --name pwc
conda activate pwc
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 matplotlib tensorboard scipy opencv -c pytorch -nvidia
conda install cupy
```

# Demos
Pretrained model can be find in
```Shell
./models/pwcnet-solar-ul.pth
```

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/pwcnet-solar-ul.pth --path=datasets/Solar_demo/test
```
where file "./datasets/Solar_demo/train" includes train set of Solar Ha images, file "./datasets/Solar_demo/test" includes test set of the Solar Ha images, file "012800" includes TiO images.

# Required Data
To evaluate/train pwcnet-ul, you can find Ha or TiO datasets in
```Shell
./datasets/Solar
```
you can generate your own .flo labels by those original images

# Training
After you have prepared the optical flow dataset, you can run train_pwcnet_ablation.py:
```Shell
python train_pwcnet_ablation.py --name pwcnet-solar-ul --stage solar --validation solar --gpus 0 --num_steps 120000 -batch_size 2 --lr 0.001 --image_size 384 448 --wdecay 0.0001 --gamma=0.5
```



