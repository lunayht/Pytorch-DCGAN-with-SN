# Pytorch SNGAN
Deep Convolutional Generative Adversarial Networks with Spectral Normalization

Google colab implementation can be found [here](https://colab.research.google.com/drive/1d7yGNYeU1w2tENWceWsU_pBkJbI2OTes?usp=sharing). 

## Results
| Number of Epochs | Fake Images | FID Score |
|      :----:      |   :-----:   |   :---:   |
| 20 | | |
| 100 | ![fake_img_e100](fake_images_e100.png) | 34.74 |

### Loss Graph During Training


## References
* Pytorch DCGAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
* https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
* https://github.com/martinarjovsky/WassersteinGAN

## Papers
* Vanilla GAN by [Goodfellow et al.](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
* Deep Convolutional GAN by [Radford et al.](https://arxiv.org/pdf/1511.06434.pdf)
* Spectral Normalization GAN by [Miyato et al.](https://openreview.net/pdf?id=B1QRgziT-)
* Fréchet Inception Distance (FID) score [Heusel et al.](https://arxiv.org/pdf/1706.08500.pdf)
