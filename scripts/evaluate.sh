set -ex

model="gas"
dataroot="./datasets/font"
name="gas_test"

python evaluate.py --dataroot ${dataroot} --model ${model} --name ${name} --phase test_unknown_content --evaluate_mode content 
python evaluate.py --dataroot ${dataroot}  --model ${model} --name ${name} --phase test_unknown_content --evaluate_mode style
python evaluate.py --dataroot ${dataroot} --model ${model} --name ${name} --phase test_unknown_style --evaluate_mode content
python evaluate.py --dataroot ${dataroot}  --model ${model} --name ${name} --phase test_unknown_style --evaluate_mode style