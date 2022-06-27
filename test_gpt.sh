'''
此代码train了GPT
'''
CUDA_VISIBLE_DEVICES=1 python run_gpt_prompt.py \
    --model_name_or_path microsoft/DialoGPT-medium \
    --model_name gpt-small \
    --do_eval \
    --validation_file data/fine-tune/test.json \
    --source_prefix "dialogue: " \
    --output_dir /output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
	--eval_steps=50 \
    --logging_steps=50 \
    --num_train_epochs=10.0 \
    --learning_rate=2e-3 \
    --max_source_length=512 \
    --generation_max_length 682 \
    --text_column dialogue \
    --summary_column response \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --pre_seq_len 50 \
    --prefix_drop 0.1 \