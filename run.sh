#!/usr/bin/env bash
set -euo pipefail

# Path to the MVTec dataset (expects the "mvtec" folder next to this repo by default).
# Adjust this if your dataset lives elsewhere.
datapath="../mvtec"

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

# Train the model (stores checkpoints and metrics under results/<project>/<group>/<run_name>/).
python3 main.py "${common_opts[@]}" "${net_opts[@]}" "${distill_opts[@]}" "${dataset_opts[@]}"

# Test using the trained checkpoints, save heatmaps, and write metrics (including inference time) to CSV.
python3 main.py --test --save_segmentation_images "${common_opts[@]}" "${net_opts[@]}" "${distill_opts[@]}" "${dataset_opts[@]}"

echo "Training, testing, and artifact export complete."
echo "- Metrics CSV: ${results_dir}/${log_project}/${log_group}/${run_name}/results.csv"
echo "- Saved segmentation heatmaps: ./output"
