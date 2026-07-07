#!/usr/bin/env bash
# ============================================================================
# setup_fogworld.sh — 在 NeoVerse 同级新建独立工作目录 FogWorld
#
# 作用：把 NeoVerse 的 diffsynth 包（v0 脚本唯一的 import 依赖）软链到一个与
# NeoVerse 同级的新目录，并放入 v0 脚本，使新实验代码独立、与原仓库互不污染。
#
# 用法（在远程 Linux 上执行）：
#     bash setup_fogworld.sh
#     # 如需硬拷贝 diffsynth（完全独立、可单独改动）而非软链接：
#     COPY_DIFFSYNTH=1 bash setup_fogworld.sh
# ============================================================================
set -euo pipefail

# ---------- 0. 路径变量（按实际情况修改）----------
SRC="${SRC:-/home/beihang/wzq/NeoVerse}"          # 原 NeoVerse 仓库
DST="${DST:-/home/beihang/wzq/FogWorld}"          # 新建的同级独立工作目录
DATAROOT="${DATAROOT:-/mnt/ssd1/jzl/datasets/nuscenes}"
CKPT="${CKPT:-/mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt}"
COPY_DIFFSYNTH="${COPY_DIFFSYNTH:-0}"             # 1=硬拷贝 diffsynth，0=软链接

echo "=== setup_fogworld ==="
echo "  SRC = $SRC"
echo "  DST = $DST"

# ---------- 1. 前置检查 ----------
[ -d "$SRC/diffsynth" ] || { echo "错误：找不到 $SRC/diffsynth"; exit 1; }
[ -f "$SRC/fog_world_v0_nuscenes.py" ] || { echo "错误：找不到 $SRC/fog_world_v0_nuscenes.py（先把脚本同步到原仓库）"; exit 1; }

# ---------- 2. 建目录 ----------
mkdir -p "$DST/outputs"

# ---------- 3. 复用 NeoVerse 的 diffsynth 包 ----------
if [ -e "$DST/diffsynth" ] || [ -L "$DST/diffsynth" ]; then
    echo "  $DST/diffsynth 已存在，跳过"
elif [ "$COPY_DIFFSYNTH" = "1" ]; then
    echo "  硬拷贝 diffsynth ..."
    cp -r "$SRC/diffsynth" "$DST/diffsynth"
else
    echo "  软链接 diffsynth ..."
    ln -s "$SRC/diffsynth" "$DST/diffsynth"
fi

# ---------- 4. 放入 v0 脚本 ----------
cp "$SRC/fog_world_v0_nuscenes.py" "$DST/"

# ---------- 5. 验证 ----------
echo "=== 验证 ==="
cd "$DST"
python -c "from diffsynth.models import ModelManager; from diffsynth.auxiliary_models.worldmirror.utils.save_utils import save_gs_ply; print('  diffsynth OK')" \
    || echo "  [警告] diffsynth 导入失败，检查 conda 环境"
[ -d "$DATAROOT/v1.0-mini" ] && echo "  nuscenes OK ($DATAROOT)" || echo "  [警告] 找不到 $DATAROOT/v1.0-mini"
[ -f "$CKPT" ] && echo "  ckpt OK ($CKPT)" || echo "  [警告] 找不到权重 $CKPT"
python -c "import nuscenes" 2>/dev/null && echo "  nuscenes-devkit OK" \
    || echo "  [需要安装] python -m pip install nuscenes-devkit  (务必用当前解释器自己的 pip)"

echo "=== 完成。下一步： ==="
echo "  cd $DST"
echo "  python fog_world_v0_nuscenes.py --version v1.0-mini --scene_index 0 --output_ply outputs/v0_baseline.ply"
echo "  python fog_world_v0_nuscenes.py --version v1.0-mini --scene_index 0 --use_gt --output_ply outputs/v0_gtcam.ply"
