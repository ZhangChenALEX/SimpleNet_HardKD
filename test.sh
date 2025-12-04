#!/bin/bash
#SBATCH -p l20-gpu                # 使用 L20 GPU 队列
#SBATCH --nodelist=dkucc-core-gpu-06   # 指定节点 06（关键行）
#SBATCH --gres=gpu:1              # 申请 1 块 GPU
#SBATCH --gres-flags=enforce-binding
#SBATCH -c 4                      # 4 CPU cores
#SBATCH --mem=32G                 # 32G 内存
#SBATCH -t 48:00:00               # 最长运行 48 小时
#SBATCH -o slurm-%j.out           # 标准输出日志
#SBATCH -e slurm-%j.err           # 错误输出日志


# Optional overrides: you can pass custom paths as positional arguments,
# e.g. `bash test.sh /path/to/mvtec /path/to/results run42`.
datapath="${1:-../mvtec}"
results_dir="${2:-results}"
run_name="${3:-run}"

# These rarely need changing, but can be overridden with env vars if desired.
log_project="${LOG_PROJECT:-MVTecAD_Results}"
log_group="${LOG_GROUP:-simplenet_mvtec}"

gpu=0
seed=0
student_backbone="wideresnet50"
teacher_backbone="wideresnet50"

dataset_name="mvtec_${mvtec_class}"
ckpt_filename="${CKPT_BASENAME:-ckpt.pth}"
ckpt_path="${results_dir}/${log_project}/${log_group}/${run_name}/models/0/${dataset_name}/${ckpt_filename}"

echo "[Test] Using dataset at: ${datapath}"
echo "[Test] Expecting checkpoint: ${ckpt_path}"
if [[ ! -f "${ckpt_path}" ]]; then
  echo "[Error] Pretrained weights not found. Please set run_name/results_dir/log_project/log_group to match your training run."
  exit 1
fi

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
  -d "${mvtec_class}"
  mvtec "${datapath}"
)

# Test using the previously trained checkpoints, save heatmaps/masks, and
# optionally add --visual_report to emit the per-sample analysis bundle.
python3 main.py --test --save_segmentation_images \
  "${common_opts[@]}" "${net_opts[@]}" "${distill_opts[@]}" "${dataset_opts[@]}"

echo "Testing complete."
echo "- Metrics CSV: ${results_dir}/${log_project}/${log_group}/${run_name}/results.csv"
echo "- Saved segmentation heatmaps: ./output"
echo "- Visual report (enable by adding --visual_report inside the python command): ./analysis"
