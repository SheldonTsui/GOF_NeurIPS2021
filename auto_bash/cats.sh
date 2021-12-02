CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --curriculum CATS \
    --output_dir output/Cats \
    --sample_interval 500 \
    --model_save_interval 2000 \
    --eval_freq 2000 \
    --load_dict \
    --set_step 2000 \
    --load_dir CKPT_TO_LOAD 
