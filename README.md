# GAS-Next

This is a group project repo for 11785 Fall 2022 at CMU. And we are still cleaning the code :)

## Requirements

* Linux
* CPU or NVIDIA GPU + CUDA CuDNN
* Python 3.8.3
* torch>=0.4.1
* torchvision>=0.2.1
* dominate>=2.3.1
* visdom>=0.1.8.3

## How to use

* Download the dataset by running

```bash
bash ./scripts/download_dataset.sh
```

* Train the model

```bash
bash ./scripts/train.sh
```

* Test the model

```bash
bash ./scripts/test.sh
```

* Evaluate the trained model

```bash
bash ./scripts/evaluate.sh
```

## Acknowledgements

We used the framework from [bicycle-gan](https://github.com/junyanz/BicycleGAN) .

Code derived and refactored from:

* [FTransGAN](https://github.com/ligoudaner377/font_translator_gan)
* [AGIS-Net](https://github.com/hologerry/AGIS-Net)
