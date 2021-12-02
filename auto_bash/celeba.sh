CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --curriculum CelebA \
    --output_dir output/celeba \
    --sample_interval 1000 \
    --model_save_interval 5000 \
    --eval_freq 5000 \
    --load_dict \
    --set_step 5000 \
    --load_dir CKPT_TO_LOAD 
