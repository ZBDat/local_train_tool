@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%train_rtdetr.py" ^
  --dataset-root "%SCRIPT_DIR%dataset" ^
  --class-names object ^
  --model coco-rtdetr-l ^
  --epochs 100 ^
  --batch 16 ^
  --imgsz 640 ^
  --augment-copies 1 ^
  --augment-seed 42 ^
  --augment-mosaic-prob 0.35 ^
  --augment-translate-scale-prob 0.6 ^
  --augment-cutout-prob 0.45 ^
  --augment-clahe-prob 0.45 ^
  --augment-gamma-prob 0.5 ^
  --augment-hist-perturb-prob 0.5 ^
  --augment-blur-noise-combo-prob 0.35 ^
  --normalize-mode per_image ^
  --project runs/detect ^
  --name rtdetr_train ^
  %*

endlocal
