python predict.py \
    --config ../user_data/checkpoints/release/config.yml \
    --gpu-ids 0 \
    --load-pthpath ../user_data/checkpoints/release/checkpoint.pth \
    --save-zippath ../prediction_result/submit.zip \
    --overfit
