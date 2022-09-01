python src/main.py \
    --generate \
    --test_data data/test_dynamic_persona.json \
    --output_path results/lightning_logs/version_0/gen.json  \
    --ckpt_path results/ckpt/epoch=4-step=29944.ckpt \
    --config_path gpt2 \
    --batch_size 32 \
    --gpu $1