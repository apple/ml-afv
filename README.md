# Adversarial Fisher Vectors for Unsupervised Representation Learning 

This software project accompanies the research paper, [Adversarial Fisher Vectors for Unsupervised Representation Learning](https://arxiv.org/abs/1910.13101).

We include sample code that can be used to train a GAN/EBM optionally with the MCMC inspired objective, and compute the Adversarial Fisher Vectors for linear classification.
## Citation
```
@article{zhai2019adversarial,
  title={Adversarial Fisher Vectors for Unsupervised Representation Learning},
  author={Zhai, Shuangfei and Talbott, Walter and Guestrin, Carlos and Susskind, Joshua M},
  booktitle={Advances in neural information processing systems},
  year={2019}
}
```
## Adversarial Fisher Vectors 
Adversarial Fisher Vectors (AVFs) provide a way of utilizing a trained GAN by extracting representations from it. AFVs achieve this by adopting an EBM view of a common GAN implementation, and represent an example with the derived Fisher Score, nomalized by the Fisher Information. In this repo, we demonstrate the use of AFVs by providing sample code for traing a GAN and linear classification on CIFAR10 with the induced representation. We also provide pretrained weights of a GAN on the combined CIFAR10 and CIFAR100 dataset, which yields state-of-the-art linear classification results on the two datasets.

## Setup
This code is written in Pytorch with Python 2.7. It's tested on Ubuntu 16.04 and CUDA 8.0 (but later versions should work too), and requires one GPU card. Run the following command to install all the dependencies:
```
pip install -r requirements.txt
```

## Getting Started 
### GAN Training (optional)
In order to compute the AFVs, the first step is to train a GAN (and interpret it as an EBM) on CIFAR10. A model with default setting can be trained by running:
```
python main.py
```
One can also skip this step by using the pretrained model found under the checkpoints directory.
### Linear Classifier
After a model is trained, we can load the a checkpoint (e.g., the one from the last iteration) and use it for training a linear classifier. This can be done by running:
```
python classifiy_cifar10.py --netG [path-to-generator-ckpt] --netD [path-to-discriminator-ckpt]
```
If using the pretrained model, this corresponds to:
```
python classifiy_cifar10.py --netG checkpoints/netG_pretrained.pth --netD checkpoints/netD_pretrained.pth
```
This will train a linear SVM classifier with dropout on the induced AFVs. Using the pretrained weights, this should give you a test accuracy of ~0.89. Note that this step is pretty time consuming, as the AFVs are of very high dimensionality and need to be generated online. 

## License
This sample code is released under the [LICENSE](LICENSE) terms.
