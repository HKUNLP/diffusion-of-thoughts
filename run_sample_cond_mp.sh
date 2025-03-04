steps_list=(4 8 16 32 64)
for steps in "${steps_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_run_sample_cond_mp.py \
        --model_path <PATH_TO_MODEL_DIR> \
        --dataset gsm8k \
        --steps $steps \
        --cot_steps 10 \
        --batch_size 16
    echo "steps: $steps done"
done

