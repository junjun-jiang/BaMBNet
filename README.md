# BaMBNet-Pytorch
Paper:\
https://arxiv.org/abs/2105.14766 (arXiv) \
https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2022.105563 (IEEE/CAA JAS) 

The code is for the work:

```
@article{liang2021BaMBNet,
  title={BaMBNet: A Blur-aware Multi-branch Network for Defocus Deblurring},
  author={Pengwei Liang, Junjun Jiang, Xianming Liu, and Jiayi Ma},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={},
  number={},
  pages={},
  year={2022},
}
```

## Requirements

``` python
pytorch == 1.7.1
kornia == 0.4.1
opencv == 4.4.0
```

### Dataset

Please refer to the official repo at [Defocus deblurring using dual-pixel data](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel).

Note that the image list of small training dataset used in meta-learning can be found in [Google Drive](https://drive.google.com/drive/folders/1OXHu8Eb9V5C0kT6R5Yzr712JCZ37BoEN?usp=sharing).

The deblurring images of DPDBlur dataset are available at [Google Drive](https://drive.google.com/file/d/1xkRaaJbqH6R0Mv42Rea3nB4hkRmgodbj/view?usp=sharing)

## Train

+ Step 1: train COC network to estimate the blur amounts of DP data.

  ```python
  python blur_train.py -opt option/train/COC_Dataset_Train.yaml
  ```

+ Step 2: prepare the COC maps for deblurring training.

  ```python
  python blur_test.py -opt option/test/COC_Dataset_Test.yaml
  ```

+ Step 3: train the deblurred network.

  ```python
  python train.py -opt option/train/Deblur_Dataset_Trained.yaml
  ```

  

## Test

```python
python test.py -opt option/test/Deblur_Dataset_Test.yaml
```

## Results

+ Results of [DPDD](https://drive.google.com/file/d/1xkRaaJbqH6R0Mv42Rea3nB4hkRmgodbj/view?usp=sharing)
+ Results of [Pixel5](https://drive.google.com/file/d/1xkRaaJbqH6R0Mv42Rea3nB4hkRmgodbj/view?usp=sharing)
+ Results of [dual_pixel_defocus_estimation_deblurring](https://drive.google.com/file/d/1xkRaaJbqH6R0Mv42Rea3nB4hkRmgodbj/view?usp=sharing)


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/facebookresearch/moco/blob/master/LICENSE) for details.

