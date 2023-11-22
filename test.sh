model_dir=dataseta
beam_size=10
batch_size=128
source_length=150
target_length=30
output_dir=model/$model_dir
dev_file=eval
test_file=test
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size