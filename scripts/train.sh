set -ex
python train.py --dataroot ./datasets/font --model gas --name gas_test --no_dropout
# python train.py --dataroot ./datasets/font --model emd --name test_new_dataset --no_dropout