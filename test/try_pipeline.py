user_prompt="A beautiful black hair girl with her eyes closed in champagne long sleeved formal dress standing in her bright room with delicate blue vases with pink roses on the left and some white roses, filled with upgraded growth all around on the right."
height=1024
width=1024
use_base=True
base_ratio=0.0
cfg=7
steps=50

intention="""Create an invitation for a Valentine's Day party to be held on February 14, 2022, at 123 Anywhere St., Any City from 7 PM until the end of the event."""
# layer_data='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/try_glyph.json'
config_dir="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/TypoClipSDXL/configs/2024_03_06_byt5_sdxl/sdxl-lora-128_noise-offset_train-byt5-mapper_frominit_mix_textseg_sdxl_canva_w8-1-1_5ep_4x16_premiumx2.py"
ckpt_dir="/pyy/openseg_blob/weicong/llm/TypoClipSDXL/work_dirs/sdxl-lora-128_noise-offset_train-byt5-mapper_frominit_mix_sdxl_canva_w1-1_8ep_4x16_premiumx2/checkpoint-1810"
model_name="albedobaseXL_v20.safetensors"
output_dir="try_pipeline"
load_typo_sdxl_pretrain_ckpt="/pyy/openseg_blob/liuzeyu/diffusion_design_if/TypoCLIPSDXL_new/work_dirs/0223-sdxl-lora-128_noise-offset_frominit_clip-pretraind-byt5_mapper_train_tlr2-wd0x2_glyph-w-10_canva-mix500k_10ep_4x16_premiumx2/checkpoint-19000"
