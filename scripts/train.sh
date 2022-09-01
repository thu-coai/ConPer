python src/main.py \
    --train_data data/train_dynamic_persona.json \
    --valid_data data/valid_dynamic_persona.json \
    --config_path gpt2 \
    --epoch 5 \
    --batch_size 4 \
    --accumulate_grad 2 \
    --gpu $1 \
    --save_dir results