# Perspective + caption → DiT360 I2P (bounded first sample)

這是 thin geometry adapter，不是作者原生 T2P 效能重現。固定上游
`3779fe7965473f6824994c663a0ae7a76bc7aafa`，保留原 native inversion/editing
演算法，只補必要 RNG、mask packing 和可顯式設定的 latent transfer 閾值。

## 環境（S3063 / pierce，個人 NAS，不碰 shared base / LayerPano）

```sh
. "$HOME/.config/s3063/environment.sh" || exit 75
/opt/miniforge/bin/conda create -n dit360-blackwell python=3.12
PY=/mnt/NAS/pierce/.workstation/S3063/conda/envs/dit360-blackwell/bin/python
$PY -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -r environment/dit360-blackwell.lock.txt
$PY -m pip check
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -m unittest discover -s tests -v
```

`environment/dit360-blackwell.conda-explicit.txt` 保存實際 Python 3.12.16 conda
builds，pip lock 來自實際 S3063 environment；唯一正規化是 conda packaging 的
build-machine file:// URL 換成已安裝的 `packaging==26.3`。Torch CUDA wheel 要用
cu128 index；lock 不是通用跨平台 lock，不宣稱 bit-exact 跨硬體。diffusers 固定
PyPI release 0.35.1，替代原 requirements 未鎖 Git proxy HEAD；CPU tiny Flux
雙 block + 原生 PersonalizeAnything processors 的 >tau/<tau forward 已測。
不安裝 training-only deepspeed/lightning/datasets，不修改原 editing.py。

## 合法模型資產

base 固定 `black-forest-labs/FLUX.1-dev` revision
`45d95c497444bbaebe766e1528792fc9158c0c3a`；LoRA 固定
`Insta360-Research/DiT360-Panorama-Image-Generation` revision
`88b5915693607d3377e1b5385d7abbe6c671d6ce`。
FLUX dev non-commercial terms 仍適用，LoRA MIT 不覆蓋 base；未驗證下游 QA 訓練授權。
必須使用獲授權者已配置的 Hub auth，不在 chat/board 傳 token，不繞 gate。

2026-11-23 PM 已以標準 Hub HEAD 確認合法 access，解除先前 401。
`$PY scripts/download_i2p_assets.py` 只抓 pinned diffusers 必需元件與 LoRA，不抓根目錄
重複 monolithic checkpoints、不下載任何其他方法生成圖。SHA256 逐檔比對 Hub LFS
expected digest，完整記錄在個人 workstation 的 `model-files.json` 與 DONE/FAILED。
下載 script 為本卡專用固定路徑，失敗保留 cache/log，不自動再登入或讀取憑證值。
CLI generation 自身 `local_files_only=True`，不會臨時下載或把 cache miss 當新 revision。

## CPU preparation（真的能跑；不是生成）

```sh
cd /mnt/NAS/E3DQA_project/tools/DiT360
$PY i2p.py \
  --image /mnt/NAS/E3DQA_project/curated_set/photorealistic/0106.jpg \
  --caption 'In the Cozy Living Room, a plush sofa sits invitingly beside a well-arranged coffee table adorned with flowers and books.' \
  --expected-sha256 a1b93c357a49bf25a42aaf592ebb2d327136bde845b38d2e6a8022ba98212c89 \
  --hfov-deg 80 --seed 42 \
  --output-dir /mnt/NAS/E3DQA_project/assets/panoramas/dit360/prepared-NEW-UNIQUE-NAME
```

一次只接受一張，不掃 metadata 或 batch。output directory 原子 mkdir，不覆蓋既有 run。
source bytes 原樣複製並記 hash/完整 decode；未知中灰 RGB128、known 白255，沒有 soft mask、
feather/erosion、後貼圖、SR、黑色對照或調參。若有真實 K，可用 `--K-json` 替代
`--hfov-deg`（互斥，3×3 zero-skew pinhole matrix，必須對應 source 原尺寸）。
首樣無已知 calibration，明標 nominal horizontal80，fx=fy=W/(2tan40°)，
principal point=((W-1)/2,(H-1)/2)，vertical FoV=2atan(H/(2fy))；不冒稱真 K。

ERP pixel centers；世界與相機 x右 y下 z前。ray=(cosφ sinλ,-sinφ,cosφ cosλ)。
source 採樣只限 z>0 且 x∈[0,W-1], y∈[0,H-1]；越界一律未知，無 edge extension。
mask 先 nearest 降到64×128 packed tokens，左右循環欄 `[last,body,first]`，
8320×1 row-major。輸出 native token crop 後 2048×1024，不再裁 RGB 左右。

## 真正單張 generation

先 CPU tests、逐檔資產核對、GPU ownership/headroom 協調。預設 wrapper 有 foreign GPU
job 就拒絕；沒有任何自動等待／搶占／kill。即使有約75GiB free，不代表別人的需求已被預約。
只有完成 operator 協調才可加 `--coordinated-with-foreign-jobs`，此旗標本身不是批准。

```sh
# 專用 tmux 新 session/window，不 send-keys 到既有 busy pane。
# 完成 coordination 後執行（不可為了繞檢查自行加旗標）：
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY scripts/run_single_i2p.py
```

wrapper 固定 `assets/panoramas/dit360/first-sample-t_db92abb3`，拒絕既有 run。
workstation `dit360-t_db92abb3/launch-TIMESTAMP/` 保存 inference.log、launch.json、
vram.jsonl、execution.json、DONE/FAILED；小型 log/summary 也複製進輸出資料夾。
每秒 nvidia-smi sampled own/whole-device peak 與 PyTorch allocator peak allocated/reserved
分開報；取樣可能漏尖峰，whole-device含他人工作，不可混為本模型 allocation。

固定 seed42，inversion/generation 兩條獨立同 seed CUDA generator；inversion 依序控制
VAE posterior 和 y_1。inversion50/generation50、gamma1、tau50、t_l0.5、guidance3.5，
source_prompt空字串/source_guidance0、eta1/strength1、start0/stop0.99、text length512。
兩支 prompt 一致：原 caption 精確加 `This is a panorama image. `。native guidance 變數2.8
在原 demo 未傳入，真有效預設是3.5。dtype float16，未引入 quantization/offload 或替代 denoiser。
每次 invert 前用原 processors，generation 後 finally 恢復；未來 loop 要在外面載入 pipe，
每 sample 重新 seed，不能重用 mutable streams 或遺留 batch2 processor 給 batch1 inversion。

## 驗證與限制

CPU suite 包含 analytic FoV/aspect/bounds、非對稱 K、yaw/seam/poles、coordinate-ramp roundtrip、
未知填色/白known polarity、padding/flatten/crop、不合法參數、no-overwrite、
VAE與invert generator動態傳遞、兩 sample processor復原、tiny真Flux雙block CPU forward。
它們不是 checkpoint GPU inference 證據。`PREPARED_NOT_GENERATED` 預覽只驗幾何；
若未有 raw_erp.png/DONE，不能宣稱首樣成功。

生成後完整 decode/尺寸2048×1024與 SHA256，原樣保留 raw_erp.png。視覺以同 K source-view、
yaw±90、seam180、zenith/nadir（longitude periodic bilinear sampling）檢查；source-view
nominal consistency 不是相機真值。軟體完成與視覺可用分開，差結果不重挑seed或重跑救圖。
本卡最多一張 curated output，不擴140、不做 downstream 3D／QA training。
