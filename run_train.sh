
## dot
# python train.py --digit --fix_src --dataset gsm8k --steps 120000 --weights_path plaid1b_weights

## dot with scheduled sampling 
#python train.py --digit --fix_src --dataset gsm8k --steps 120000 --weights_path plaid1b_weights --min_prob 0.95

## mp-dot
# python train.py --digit --fix_src --cot --dataset gsm8k --steps 31000 --weights_path plaid1b_weights 

## mp-dot with glance sampling 
# python train.py --digit --fix_src --cot --dataset gsm8k --steps 31000 --weights_path plaid1b_weights  --glance

## continue pretraining
# python train.py --digit --dataset gsm8k --steps 120000 --weights_path plaid1b_weights
