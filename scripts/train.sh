set -ex
python train.py --dataroot ./datasets/font --model font_translator_gan --name ftransgan_base --no_dropout
# python train.py --dataroot ./datasets/font --model emd --name test_new_dataset --no_dropout