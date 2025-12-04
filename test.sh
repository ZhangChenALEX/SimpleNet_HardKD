#!/bin/bash
#SBATCH -p l20-gpu
#SBATCH --nodelist=dkucc-core-gpu-06
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

############################
# 1. 读取输入参数
############################
mvtec_class="${1:-capsule}"      # 默认 capsule，可传 carpet/bottle 等
datapath="${1:-../mvtec}"
results_dir="${2:-results}"
run_name="${3:-run}"

log_project="MVTecAD_Results"
log_group="simplenet_mvtec"

echo "[Test] Class: ${mvtec_class}"
echo "[Test] Dataset: ${datapath}"
echo "[Test] Results dir: ${results_dir}"

############################
# 2. checkpoint 路径
############################
ckpt_path="${results_dir}/${log_project}/${log_group}/${run_name}/models/0/mvtec_${mvtec_class}/ckpt.pth"

echo "[Test] Expecting checkpoint at:"
echo "       ${ckpt_path}"

if [[ ! -f "${ckpt_path}" ]]; then
    echo "[Error] Could not find checkpoint!"
    echo "Path was:"
    echo "  ${ckpt_path}"
    echo "Please make sure mvtec_${mvtec_class} was trained."
    exit 1
fi

############################
# 3. python 运行参数
############################
common_opts=(
  --gpu 0
  --seed 0
  --log_group "${log_group}"
  --log_project "${log_project}"
  --results_path "${results_dir}"
  --run_name "${run_name}"
)

net_opts=(
  net
  -b wideresnet50
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
  --teacher_backbone wideresnet50
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
  -d "${mvtec_class}"
  mvtec "${datapath}"
)

############################
# 4. Run test + visual report
############################
echo "[Test] Running evaluation and visual report generation..."

python3 main.py --test --save_segmentation_images --visual_report \
  "${common_opts[@]}" "${net_opts[@]}" "${distill_opts[@]}" "${dataset_opts[@]}"

echo "======================================="
echo "Testing complete."
echo "Metrics CSV:"
echo "  ${results_dir}/${log_project}/${log_group}/${run_name}/results.csv"
echo ""
echo "Heatmaps saved to:"
echo "  ./output/mvtec_${mvtec_class}/"
echo ""
echo "Visual report saved to:"
echo "  ./analysis/mvtec_${mvtec_class}/"
echo "======================================="
