# CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 \
#     --master_port=26662 train_scripts/train_controlnet.py \
#     configs/pixart_app_config/PixArt_xl2_img1024_controlHed_Half.py \
#     --work-dir output/debug
python3 ./train_scripts/train_pixart_lora_hf.py --dataset_name ./pokemon-blip-captions
python3 ./train_scripts/train_pixart_lora_hf.py --dataset_name imagefolder --train_data_dir ./data/flickr1k_local --image_column image --caption_column text