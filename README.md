# U-Net X-ray Segmentation

Binary segmentation of X-ray images with a U-Net, implemented in PyTorch.
150 X-ray images: 100 manually annotated (80 train / 20 validation) and 50 unlabeled test images.



## License

**Code:** MIT License, see [LICENSE](LICENSE).

**Model:** The architecture follows Ronneberger, Fischer and Brox, *U-Net: Convolutional Networks for
Biomedical Image Segmentation* (2015), [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
The network is implemented from scratch in `model.py` and trained from random initialization on the course
dataset. No pretrained weights and no third-party model code are used, so no third-party model license applies.
The trained weights are released under the same MIT License as the code.