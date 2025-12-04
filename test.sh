#!/usr/bin/env bash
set -euo pipefail

# Path to the MVTec dataset (expects the "mvtec" folder next to this repo by default).
# Update this if your dataset lives elsewhere.
datapath="../mvtec"

# Existing results folder that already contains trained checkpoints.
# Set run_name to the same value you used during training so the script
# can find models/<idx>/<dataset>/models.ckpt under results_dir/log_project/log_group/run_name/.
results_dir="results"
log_project="MVTecAD_Results"
log_group="simplenet_mvtec"
run_name="run"

gpu=0
seed=0
student_backbone="wideresnet50"
teacher_backbone="wideresnet50"

common_opts=(
  --gpu "${gpu}"
  --seed "${seed}"
  --log_group "${log_group}"
  --log_project "${log_project}"
  --results_path "${results_dir}"
  --run_name "${run_name}"
)

net_opts=(
  net
  -b "${student_backbone}"
  -le layer2
  -le layer3
  --pretrain_embed_dimension 1536
  --target_embed_dimension 1536
  --patchsize 3
  --meta_epochs 40
  --embedding_size 256
  --gan_epochs 4
  --noise_std 0.015
  --dsc_hidden 1024
  --dsc_layers 2
  --dsc_margin .5
  --pre_proj 1
)

distill_opts=(
  --use_distillation
  --teacher_backbone "${teacher_backbone}"
  --distill_weight 0.1
  --distill_max_patches 192
  --distill_knn 5
  --distill_direction_weight 1.0
  --distill_distance_weight 1.0
  --distill_neighborhood_weight 1.0
  --distill_embedding_weight 0.5
  --distill_tightness_weight 0.25
)

dataset_opts=(
  dataset
  --batch_size 8
  --resize 329
  --imagesize 288
  mvtec "${datapath}"
)

# Test using the previously trained checkpoints, save heatmaps/masks, and
# optionally add --visual_report to emit the per-sample analysis bundle.
python3 main.py --test --save_segmentation_images \
  "${common_opts[@]}" "${net_opts[@]}" "${distill_opts[@]}" "${dataset_opts[@]}"

echo "Testing complete."
echo "- Metrics CSV: ${results_dir}/${log_project}/${log_group}/${run_name}/results.csv"
echo "- Saved segmentation heatmaps: ./output"
