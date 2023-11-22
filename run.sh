model_dir=dataseta
lr=5e-5
batch_size=64
beam_size=10
source_length=150
target_length=30
data_dir=./data/CodeSearchNet
output_dir=model/$model_dir
train_file=train
dev_file=eval
eval_steps=1000 
train_steps=300000
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps
