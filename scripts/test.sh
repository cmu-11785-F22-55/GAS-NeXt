set -ex

model="gas"
dataroot="./datasets/font"
name="gas_test"
dataset_mode="font"

python test.py --dataroot ${dataroot}  --model ${model} --dataset_mode ${dataset_mode} --name ${name} --phase test_unknown_style  --eval --no_dropout
python test.py --dataroot ${dataroot}  --model ${model} --dataset_mode ${dataset_mode} --name ${name} --phase test_unknown_content  --eval --no_dropout