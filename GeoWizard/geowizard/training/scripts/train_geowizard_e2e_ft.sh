# accelerate config
root_path='/home/hmi/Downloads/renders/'
csv_train_path='/mnt/hmi/thuong/Photoface_dist/geowizard_photoface_TrainValTest/dataset_0/train.csv'
csv_valid_path='/mnt/hmi/thuong/Photoface_dist/geowizard_photoface_TrainValTest/dataset_0/val.csv'
output_dir='/media/hmi/Transcend/photoface_ft_checkpoints/training_logs'
output_valid_dir='/media/hmi/Transcend/photoface_ft_checkpoints/validation_results'
pretrained_model_name_or_path="lemonaddie/geowizard"
fined_tune_from_checkpoint='lemonaddie/geowizard'
train_batch_size=1
gradient_accumulation_steps=16
num_train_epochs=10
checkpointing_steps=2500
learning_rate=3e-5
lr_warmup_steps=0
dataloader_num_workers=16
dataset_name='photoface'
seed=1234

accelerate launch --config_file ../node_config/1gpu.yaml \
                ../training/train_depth_normal.py \
                --pretrained_model_name_or_path $pretrained_model_name_or_path \
                --fined_tune_from_checkpoint $fined_tune_from_checkpoint \
                --csv_train_path $csv_train_path \
                --csv_valid_path $csv_valid_path \
                --dataset_path $root_path  \
                --dataset_name $dataset_name \
                --output_dir $output_dir \
                --output_valid_dir $output_valid_dir \
                --e2e_ft \
                --noise_type="zeros" \
                --max_train_steps 20000 \
                --checkpointing_steps $checkpointing_steps \
                --train_batch_size $train_batch_size \
                --gradient_accumulation_steps $gradient_accumulation_steps \
                --learning_rate $learning_rate \
                --dataloader_num_workers $dataloader_num_workers \
                --lr_total_iter_length 20000 \
                --lr_warmup_steps 100 \
                --seed $seed \
                --output_dir $output_dir \
                --enable_xformers_memory_efficient_attention \
                --use_8bit_adam \
                --gradient_checkpointing \
                --use_ema
