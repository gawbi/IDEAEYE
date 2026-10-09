# Experiments summary

| experiment | seed | git | reproducible | headline |
|---|---|---|---|---|
| 20260924_103010_det_yolo11n_MONO | 42 | no-git | NO | mAP50=0.6699, egg_loc_recall=0.9841, infer_ms=1.8583 |
| 20260924_104909_det_yolo11n_COLOR | 42 | no-git | NO | mAP50=0.5816, egg_loc_recall=0.9969, infer_ms=2.3981 |
| 20260924_111044_cls_mono_effb0 | 42 | no-git | NO | acc=0.6314, macro_recall=0.6741, normal_fpr@crack_recall95=0.3567, recall_2/6=0.8100, recall_2/8=0.5729, recall_3/1=0.6190 |
| 20260924_121017_cls_color_effb0 | 42 | no-git | NO | acc=0.9662, macro_recall=0.6219, normal_fpr@crack_recall95=0.0550 |
| 20260924_124523_patchcore_mono | 42 | no-git | NO | auroc=0.6961, normal_fpr@crack_recall95=0.7225 |
| 20260924_142357_candling_threshold_baseline | 42 | no-git | NO | auroc_blood_vs_normal=0.6283, auroc_blood_vs_dark=0.4318, normal_fpr@blood_recall88=0.6850 |
| 20260924_142411_abl_mono_res320 | 42 | no-git | NO | acc=0.6333, macro_recall=0.6844, normal_fpr@crack_recall95=0.3356, recall_2/6=0.7975, recall_2/8=0.5564, recall_3/1=0.6587 |
| 20260924_155235_abl_mono_clahe | 42 | no-git | NO | acc=0.6134, macro_recall=0.6622, normal_fpr@crack_recall95=0.4219, recall_2/6=0.8037, recall_2/8=0.5513, recall_3/1=0.5635 |
| 20260924_165645_abl_mono_wfocal | 42 | no-git | NO | acc=0.6247, macro_recall=0.6657, normal_fpr@crack_recall95=0.2943, recall_2/6=0.7819, recall_2/8=0.5387, recall_3/1=0.5476 |
| 20260924_175728_cls_candling_none | 42 | no-git | NO | acc=0.8572, macro_recall=0.8482, recall_3/1=0.6349 |
| 20260924_181429_cls_candling_clahe | 42 | no-git | NO | acc=0.8537, macro_recall=0.8439, recall_3/1=0.6111 |
| 20260924_183232_cls_candling_unsharp | 42 | no-git | NO | acc=0.8554, macro_recall=0.8427, recall_3/1=0.6190 |
| 20260924_185030_benchmark_MONO | 0 | no-git | NO | per_egg_ms_p50=6.4791, total_ms_p95=120.5885 |
| 20260924_185711_cls_mono_state_effb0 | 42 | no-git | NO | acc=0.7329, macro_recall=0.7300, normal_fpr@crack_recall95=0.3871 |
| 20260925_165057_cls_mono_crackbin_effb0 | 42 | no-git | NO | acc=0.9243, macro_recall=0.9270, normal_fpr@crack_recall95=0.0958 |
| 20260925_174341_cls_color_washsplit_effb0 | 42 | no-git | NO | acc=0.9323, macro_recall=0.9337, normal_fpr@crack_recall95=0.0017 |
| 20260925_181741_abl_mono_combined | 42 | no-git | NO | acc=0.6270, macro_recall=0.6765, normal_fpr@crack_recall95=0.2786, recall_2/6=0.8100, recall_2/8=0.5577, recall_3/1=0.5873 |
| 20260925_195844_benchmark_MONO | 0 | no-git | NO | per_egg_ms_p50=8.2208, total_ms_p95=134.0345 |
| 20260925_195910_cls_mono_3cls_effb0 | 42 | no-git | NO | acc=0.8295, macro_recall=0.8116, normal_fpr@crack_recall95=0.2250 |
| 20260925_213330_benchmark_MONO_onnx | 0 | no-git | NO | per_egg_ms_p50=7.0942, total_ms_p95=133.5003 |
| 20260925_213354_benchmark_MONO_torch | 0 | no-git | NO | per_egg_ms_p50=6.1424, total_ms_p95=112.0446 |
| 20260925_213520_benchmark_MONO_torch | 0 | no-git | NO | per_egg_ms_p50=2.8339, total_ms_p95=64.3530 |
| 20260925_213808_det_yolo26n_MONO | 42 | no-git | NO | mAP50=0.6750, egg_loc_recall=0.9850, infer_ms=1.7663 |
| 20260925_221138_cls_mono_crackbin_effb2 | 42 | no-git | NO | acc=0.9298, macro_recall=0.9315, normal_fpr@crack_recall95=0.0894 |
| 20260925_232710_benchmark_MONO_torch | 0 | no-git | NO | per_egg_ms_p50=2.1400, total_ms_p95=50.5308 |
| 20260925_233038_benchmark_MONO_torch | 0 | no-git | NO | per_egg_ms_p50=1.8969, total_ms_p95=48.0627 |
| 20261009_173953_sev_mono_effb0 | 42 | 3c76bd6b47959498d136c5906510f78dc8d8a0ed-dirty | NO | acc=0.8202, macro_recall=0.7870, normal_fpr@crack_recall95=0.5568 |
