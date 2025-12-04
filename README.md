# SimpleNet

![Cover image](imgs/cover.png)

**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**

Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang

[Paper link](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)

## Introduction
This repository contains a PyTorch reimplementation of **SimpleNet**, a defect detection and localization network composed of a feature encoder, feature generator, and defect discriminator. The goal is to keep the architecture and training recipe simple while delivering strong anomaly detection and segmentation performance.

## Environment
- Python **3.8** (other versions may work)
- PyTorch **1.12.1**
- TorchVision **0.13.1**
- NumPy **1.22.4**
- OpenCV-Python **4.5.1**

Install the requirements with your preferred environment manager (e.g., `conda`, `pip`).

## Dataset
- Default target: **MvTecAD**. Download from the [official site](https://www.mvtec.com/company/research/datasets/mvtec-ad/).
- Expected layout: keep the original folder structure. By default `run.sh` assumes the dataset sits next to this repo in a sibling folder named `mvtec` (i.e., `../mvtec`).
- To use a different location or subset of classes, edit `datapath` or `classes` in `run.sh`, or pass the corresponding CLI flags to `main.py`.
  - If you want a quick per-image visual report (heatmap overlay, GT comparison, FPR histogram, median-threshold FP/FN list), add the `--visual_report` flag to the testing command. Use `--visual_report_dir` to pick an output folder (default: `analysis`) and `--visual_report_index` to choose which test sample to visualize (0-based index).

## How to Run
The project ships with a convenience script that trains and then tests in one go.

```bash
bash run.sh
```

Key arguments inside `run.sh` (feel free to tweak):
- `datapath`: dataset root (defaults to `../mvtec`).
- `log_project`, `log_group`, `run_name`, `results_dir`: control where logs and metrics are stored.
- `meta_epochs`: number of training epochs (set to 40 by default).
- `--save_segmentation_images`: enabled so heatmaps/masks are exported during testing.

You can also invoke the CLI directly for custom runs, e.g.:
```bash
python3 main.py --dataset mvtec --datapath ../mvtec --meta_epochs 40
python3 main.py --dataset mvtec --datapath ../mvtec --test --save_segmentation_images
```

For **test only**, run:

```bash
bash test.sh [datapath] [results_dir] [run_name]
```

- Defaults: `datapath=../mvtec`, `results_dir=results`, `run_name=run`, `log_project=MVTecAD_Results`, `log_group=simplenet_mvtec`.
- The script expects pretrained weights at `results/<log_project>/<log_group>/<run_name>/models/0/mvtec/models.ckpt` and will abort with a clear error if the file is missing.
- Outputs remain unchanged: metrics CSV under `results/<log_project>/<log_group>/<run_name>/results.csv`, segmentation heatmaps under `./output/`, and optional visual reports under `./analysis/` (when `--visual_report` is added to the python command inside the script).

## End-to-End Workflow (English)
1. **Prepare data**: place the MvTecAD dataset at `../mvtec` or point `datapath` elsewhere.
2. **Configure run**: adjust `run.sh` variables (paths, epochs, class list) or supply equivalent CLI flags.
3. **Train**: `SimpleNet.train` iterates over `meta_epochs`, training the discriminator and evaluating on the test split each round.
4. **Test & visualize**: `SimpleNet.test` runs inference, measures per-image inference time, and optionally saves heatmaps/masks via `--save_segmentation_images`.
5. **Collect results**: metrics (image-level AUROC, pixel-level AUROC, AUPRO, average inference time in seconds per image) are aggregated into a CSV; visual outputs are written to disk.

## Where to Find Outputs
- **Metrics CSV**: `results/<log_project>/<log_group>/<run_name>/results.csv` (adjustable via `run.sh`). The last row contains the mean over all evaluated classes.
- **Heatmaps & masks**: saved under `output/` in the repository root when `--save_segmentation_images` is enabled.

## Citation
```bibtex
@inproceedings{liu2023simplenet,
  title={SimpleNet: A Simple Network for Image Anomaly Detection and Localization},
  author={Liu, Zhikang and Zhou, Yiming and Xu, Yuansheng and Wang, Zilei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20402--20411},
  year={2023}
}
```

## Acknowledgement
Thanks for great inspiration from [PatchCore](https://github.com/amazon-science/patchcore-inspection).

## License
All code within the repo is under the [MIT license](https://mit-license.org/).
