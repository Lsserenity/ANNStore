#!/usr/bin/env bash
set -euo pipefail

# 低内存前置脚本（单脚本双模式）：执行步骤 1-5
#
# 用法：
#   bash Scripts/ssd-layout/prepare_dataset_spanpool_front_lowmem.sh <dataset> [base|packed] [config]
#
# 模式：
# - base（默认）：HnswL1LayoutBase
# - packed：HnswL1LayoutPacked
#
# 关键环境变量：
# - ANNFS_LOWMEM_VARIANT=base|packed  （可替代第2个位置参数）
# - ANNFS_LOWMEM_GRAPH_MODE=auto|rebuild|reuse|resume(=reuse)
# - ANNFS_PACK_METHOD=dp_hybrid         （packed模式生效）
# - ANNFS_PACK_TARGET_BLOCK_BYTES=0|<bytes>               （packed模式生效，0=跟随设备逻辑块）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

DATASET="${1:-}"
VARIANT_ARG="${2:-}"
VARIANT="${VARIANT_ARG:-${ANNFS_LOWMEM_VARIANT:-base}}"
CONFIG_PATH="${3:-${ANNFS_CONFIG:-UseConfigs/datasets.base.json}}"

if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <dataset> [base|packed] [config]"
  exit 1
fi

if [[ "$VARIANT" != "base" && "$VARIANT" != "packed" ]]; then
  echo "[ERROR] invalid variant: $VARIANT (expect: base|packed)"
  exit 1
fi

if ! jq -e --arg ds "$DATASET" '.dataset_list[$ds] != null' "$CONFIG_PATH" >/dev/null; then
  echo "[ERROR] dataset '$DATASET' not found in $CONFIG_PATH"
  exit 1
fi

TRACE_PATH="${ANNFS_TRACE_PATH:-workload_sizes.npy}"
if [[ ! -f "$TRACE_PATH" ]]; then
  echo "[ERROR] trace file not found: $TRACE_PATH"
  exit 1
fi

if [[ "$VARIANT" == "packed" ]]; then
  WRITER_BIN="./execs/HnswL1LayoutPacked"
  LAYOUT_SUFFIX="lowmem_packed"
  MANIFEST_SUFFIX="lowmem_packed"
  ARTIFACT_PREFIX="lowmem_packed_stepA_artifacts"
else
  WRITER_BIN="./execs/HnswL1LayoutBase"
  LAYOUT_SUFFIX="lowmem"
  MANIFEST_SUFFIX="lowmem"
  ARTIFACT_PREFIX="lowmem_stepA_artifacts"
fi

if [[ ! -x "$WRITER_BIN" ]]; then
  echo "[ERROR] missing writer binary: $WRITER_BIN"
  echo "[HINT] run: ./build.sh Release"
  exit 1
fi
RELABEL_BIN="./execs/HnswRelabelIndex"
if [[ ! -x "$RELABEL_BIN" ]]; then
  echo "[ERROR] missing relabel binary: $RELABEL_BIN"
  echo "[HINT] run: ./build.sh Release"
  exit 1
fi

WRITER_CFG="/tmp/ds.${DATASET}.writer.json"
LAYOUT_JSON="/tmp/${DATASET}_layout_l1centric_blockaligned_${LAYOUT_SUFFIX}.json"
MANIFEST_JSON="/tmp/${DATASET}_manifest_l1centric_blockaligned_${MANIFEST_SUFFIX}.json"
LABEL_MAP_TMP="/tmp/label_map_${DATASET}_l1centric_blockaligned.txt"
DUMP_JSON="mapped_io_q10_${DATASET}.json"
MAIN_CFG_DIR="${ANNFS_RESULT_CONFIG_DIR:-$ROOT/result_configs}"
MAIN_CFG_FILE="$MAIN_CFG_DIR/${DATASET}.json"
DATA_ROOT=$(jq -r --arg ds "$DATASET" '.dataset_list[$ds].data_root_dir' "$CONFIG_PATH")
DEV_PATH=$(jq -r '.dev_path' "$CONFIG_PATH")
GRAPH_PREFIX="$DATA_ROOT/hnsw_l0_graph_${DATASET}_lowmem"
GRAPH_MODE="${ANNFS_LOWMEM_GRAPH_MODE:-auto}"  # auto|rebuild|reuse|resume
if [[ "$GRAPH_MODE" == "resume" ]]; then
  GRAPH_MODE="reuse"
fi
if [[ "$GRAPH_MODE" != "auto" && "$GRAPH_MODE" != "rebuild" && "$GRAPH_MODE" != "reuse" ]]; then
  echo "[ERROR] invalid ANNFS_LOWMEM_GRAPH_MODE=$GRAPH_MODE (expect auto|rebuild|reuse|resume)"
  exit 1
fi
PACK_METHOD="${ANNFS_PACK_METHOD:-dp_hybrid}"  # packed only
TARGET_BLOCK_BYTES="${ANNFS_PACK_TARGET_BLOCK_BYTES:-0}"  # packed only
ARTIFACT_MD="$DATA_ROOT/${ARTIFACT_PREFIX}_${DATASET}.md"
RELABEL_MAP_TXT="$DATA_ROOT/internal_to_packed64_${DATASET}_${LAYOUT_SUFFIX}.txt"
RELABEL_DUMP_BEFORE="$DATA_ROOT/internal_label_before_${DATASET}_${LAYOUT_SUFFIX}.txt"
RELABEL_DUMP_AFTER="$DATA_ROOT/internal_label_after_${DATASET}_${LAYOUT_SUFFIX}.txt"
RELABELED_INDEX="$DATA_ROOT/${DATASET}_l1centric_${LAYOUT_SUFFIX}_relabel.bin"
RUN_CFG="$DATA_ROOT/ds.${DATASET}.${LAYOUT_SUFFIX}.offset_relabel.json"

# 1) 切换前清理
yes y | sudo -n bash Scripts/prepare_ssd_environment.sh --disk "$DEV_PATH" --cool-time 30 || true
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
rm -f /tmp/ds.${DATASET}.*.json "/tmp/spanpool.advisor.${DATASET}.log" "/tmp/ds.spanpool.advisor.${DATASET}.json"
unset ANNFS_MAPPED_SPANPOOL_RING_DEPTH ANNFS_MAPPED_SPANPOOL_BUF1 ANNFS_MAPPED_SPANPOOL_BUF2 ANNFS_MAPPED_SPANPOOL_BUF3 ANNFS_MAPPED_SPANPOOL_BUF4 || true

# 2) 布局生成（writer 仍会产出临时 32bit label_map，但不再落盘保存）
jq '.mapped_span_mode="split"' "$CONFIG_PATH" > "$WRITER_CFG"
sudo -n mkdir -p "$DATA_ROOT"

if [[ "$VARIANT" == "packed" ]]; then
  sudo -n "$WRITER_BIN" \
    "$WRITER_CFG" \
    "$DATASET" \
    "$TRACE_PATH" \
    "$LAYOUT_JSON" \
    "$MANIFEST_JSON" \
    "$LABEL_MAP_TMP" \
    0 \
    "$GRAPH_PREFIX" \
    "$GRAPH_MODE" \
    "$TARGET_BLOCK_BYTES" \
    "$PACK_METHOD"
else
  sudo -n "$WRITER_BIN" \
    "$WRITER_CFG" \
    "$DATASET" \
    "$TRACE_PATH" \
    "$LAYOUT_JSON" \
    "$MANIFEST_JSON" \
    "$LABEL_MAP_TMP" \
    0 \
    "$GRAPH_PREFIX" \
    "$GRAPH_MODE"
fi

# 3) 产物落盘（不保存 32bit label_map）
sudo -n rm -f "$LABEL_MAP_TMP" || true
sudo -n cp -f "$LAYOUT_JSON" "$DATA_ROOT/${DATASET}_layout_l1centric_blockaligned_${LAYOUT_SUFFIX}.json"
sudo -n cp -f "$MANIFEST_JSON" "$DATA_ROOT/${DATASET}_manifest_l1centric_blockaligned_${MANIFEST_SUFFIX}.json"

# 4) 生成 internal_id -> packed64_label 映射，并 relabel 到新 index（不覆盖原索引）
# packed64: [3 bits crossed_blocks][61 bits start_byte]
python3 - "$GRAPH_PREFIX.labels.bin" "$DATA_ROOT/${DATASET}_manifest_l1centric_blockaligned_${MANIFEST_SUFFIX}.json" "$RELABEL_MAP_TXT" <<'PY'
import struct, sys
labels_bin, manifest_txt, out_txt = sys.argv[1], sys.argv[2], sys.argv[3]

mp = {}
with open(manifest_txt, "r", encoding="utf-8", errors="ignore") as f:
    for ln in f:
        if not ln.strip() or ln.startswith("#"):
            continue
        ps = ln.split()
        if len(ps) < 7:
            continue
        label = int(ps[0])
        start_byte = int(ps[1])
        crossed = int(ps[5])
        if crossed < 0 or crossed > 7:
            raise RuntimeError(f"invalid crossed_blocks={crossed} for label={label}")
        if start_byte < 0 or start_byte >= (1 << 61):
            raise RuntimeError(f"start_byte out of 61-bit range: {start_byte} label={label}")
        packed64 = (crossed << 61) | start_byte
        old = mp.get(label)
        if old is not None and old != packed64:
            raise RuntimeError(
                f"inconsistent packed64 for same label={label}: {old} vs {packed64}")
        mp[label] = packed64

data = open(labels_bin, "rb").read()
if len(data) % 8 != 0:
    raise RuntimeError(f"invalid labels.bin size: {len(data)}")
vals = struct.unpack("<" + "Q" * (len(data)//8), data)
missing = 0
seen_packed = {}
with open(out_txt, "w", encoding="utf-8") as o:
    o.write("# internal_id packed64_label\n")
    for i, lbl in enumerate(vals):
        packed = mp.get(lbl)
        if packed is None:
            missing += 1
            continue
        if packed in seen_packed:
            raise RuntimeError(
                f"duplicate packed64 label detected: {packed} "
                f"(internal_id={i}, existing_internal_id={seen_packed[packed]})")
        seen_packed[packed] = i
        o.write(f"{i} {packed}\n")
if missing:
    raise RuntimeError(
        f"missing packed64 label for {missing} labels; manifest={manifest_txt}")
PY

sudo -n "$RELABEL_BIN" \
  "$WRITER_CFG" \
  "$DATASET" \
  "$RELABEL_MAP_TXT" \
  "$RELABELED_INDEX" \
  "$RELABEL_DUMP_BEFORE" \
  "$RELABEL_DUMP_AFTER"

# 5) 生成后续实验专用配置（绑定对应模式的新 index）
# 这里采用：use_label_map=false + label_decode_mode=packed_map（packed64 label）
# 说明：relabel 后 label 语义与 GT 不一致，默认跳过 recall 统计（skip_recall=true）。
jq --arg ds "$DATASET" \
   --arg idx "$RELABELED_INDEX" \
   '.label_decode_mode="packed_map"
    | .use_label_map=false
    | .skip_recall=true
    | .mapped_span_mode="split"
    | .label_map_path=null
    | .dataset_list[$ds].hnsw_index_path=$idx
    | .dataset_list[$ds].label_map_path=null' \
   "$WRITER_CFG" > "$RUN_CFG"

# 6) offset-io 小样本校验（使用新索引配置；默认 packed label 解码）
sudo -n ./execs/HnswOffsetIO \
  "$RUN_CFG" \
  "$DATASET" \
  1 1

if [[ "$VARIANT" == "packed" ]]; then
  cat > "/tmp/${ARTIFACT_PREFIX}_${DATASET}.md" <<EOF
# ${DATASET} LowMemPacked StepA 产物说明

- \`${DATASET}_layout_l1centric_blockaligned_lowmem_packed.json\`
- \`${DATASET}_manifest_l1centric_blockaligned_lowmem_packed.json\`
- \`hnsw_l0_graph_${DATASET}_lowmem.offsets.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.neighbors.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.labels.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.l1_seeds.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.owner.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.dist.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.ordered_clusters.json\`
- \`hnsw_l0_graph_${DATASET}_lowmem.write_order.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.payload_sizes.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.meta.json\`
- \`internal_to_packed64_${DATASET}_${LAYOUT_SUFFIX}.txt\`
- \`${DATASET}_l1centric_${LAYOUT_SUFFIX}_relabel.bin\`
- \`ds.${DATASET}.${LAYOUT_SUFFIX}.offset_relabel.json\`

## 参数
- \`ANNFS_LOWMEM_GRAPH_MODE=auto|rebuild|reuse|resume(=reuse)\`
- \`ANNFS_PACK_METHOD=dp_hybrid\`
- \`ANNFS_PACK_TARGET_BLOCK_BYTES=0|<bytes>\`
EOF
else
  cat > "/tmp/${ARTIFACT_PREFIX}_${DATASET}.md" <<EOF
# ${DATASET} LowMem StepA 产物说明

- \`${DATASET}_layout_l1centric_blockaligned_lowmem.json\`
- \`${DATASET}_manifest_l1centric_blockaligned_lowmem.json\`
- \`hnsw_l0_graph_${DATASET}_lowmem.offsets.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.neighbors.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.labels.bin\`
- \`hnsw_l0_graph_${DATASET}_lowmem.meta.json\`

## 参数
- \`ANNFS_LOWMEM_GRAPH_MODE=auto|rebuild|reuse|resume(=reuse)\`
EOF
fi
sudo -n mv -f "/tmp/${ARTIFACT_PREFIX}_${DATASET}.md" "$ARTIFACT_MD"

# 5) Optional bucket advisor (not required for a functional packed layout).
if [[ -x Scripts/spanpool_bucket_advisor.sh ]]; then
  bash Scripts/spanpool_bucket_advisor.sh "$RUN_CFG" "$DATASET" ./execs/HnswMappedOffsetIO 1 1
else
  echo "[WARN] Scripts/spanpool_bucket_advisor.sh is not included; using default buckets"
fi

REC_JSON=$(python3 - "$DATASET" <<'PY'
import re
import sys
p=f"/tmp/spanpool.advisor.{sys.argv[1]}.log"
try:
    t=open(p,'r',errors='ignore').read()
except FileNotFoundError:
    print('')
    raise SystemExit
ms=re.findall(r'mapped_spanpool_buckets=(\{[^\n]*\})',t)
print(ms[-1] if ms else '')
PY
)

BASE_BUCKETS=$(python3 - "$DATASET" <<'PY'
import json,sys
ds=sys.argv[1]
mp={
  "mnist1m":{"1":256,"2":64,"4":16,"8":8,"9":8},
  "sift10m":{"1":256,"2":64,"4":16,"5":110,"6":66,"7":38},
  "glove1m":{"1":256,"2":64,"4":16,"8":8},
  "deep10m":{"1":256,"2":128,"4":48,"8":16},
  "deep100m":{"1":256,"2":128,"4":64,"8":32,"16":16},
}
print(json.dumps(mp.get(ds,{"1":256,"2":64,"4":16}), separators=(',',':')))
PY
)

if [[ -z "$REC_JSON" ]]; then
  echo "[WARN] advisor did not emit recommendation buckets; keep existing defaults"
  mkdir -p "$MAIN_CFG_DIR"
  if [[ -f "$MAIN_CFG_FILE" ]]; then
    jq --argjson b "$BASE_BUCKETS" '.buckets=$b' "$MAIN_CFG_FILE" > "${MAIN_CFG_FILE}.tmp"
  else
    jq -n --argjson b "$BASE_BUCKETS" '{buckets:$b}' > "${MAIN_CFG_FILE}.tmp"
  fi
  mv -f "${MAIN_CFG_FILE}.tmp" "$MAIN_CFG_FILE"
  echo "[INFO] saved dataset buckets(base): $MAIN_CFG_FILE(.buckets)"
else
  echo "[INFO] advisor recommendation for $DATASET: $REC_JSON"
  mkdir -p "$MAIN_CFG_DIR"
  MERGED_BUCKETS="$(jq -cn --argjson b "$BASE_BUCKETS" --argjson r "$REC_JSON" '$b * $r')"
  if [[ -f "$MAIN_CFG_FILE" ]]; then
    jq --argjson b "$MERGED_BUCKETS" '.buckets=$b' "$MAIN_CFG_FILE" > "${MAIN_CFG_FILE}.tmp"
  else
    jq -n --argjson b "$MERGED_BUCKETS" '{buckets:$b}' > "${MAIN_CFG_FILE}.tmp"
  fi
  mv -f "${MAIN_CFG_FILE}.tmp" "$MAIN_CFG_FILE"
  echo "[INFO] saved dataset buckets: $MAIN_CFG_FILE(.buckets)"
fi

echo "[DONE] dataset=$DATASET"
echo "[DONE] variant=$VARIANT"
echo "[DONE] data_root=$DATA_ROOT"
echo "[DONE] layout_json=$DATA_ROOT/${DATASET}_layout_l1centric_blockaligned_${LAYOUT_SUFFIX}.json"
echo "[DONE] manifest_json=$DATA_ROOT/${DATASET}_manifest_l1centric_blockaligned_${MANIFEST_SUFFIX}.json"
echo "[DONE] relabel_map=$RELABEL_MAP_TXT"
echo "[DONE] relabeled_index=$RELABELED_INDEX"
echo "[DONE] relabel_dump_before=$RELABEL_DUMP_BEFORE"
echo "[DONE] relabel_dump_after=$RELABEL_DUMP_AFTER"
echo "[DONE] run_cfg=$RUN_CFG"
echo "[DONE] graph_offsets=${GRAPH_PREFIX}.offsets.bin"
echo "[DONE] graph_neighbors=${GRAPH_PREFIX}.neighbors.bin"
echo "[DONE] graph_labels=${GRAPH_PREFIX}.labels.bin"
echo "[DONE] graph_meta=${GRAPH_PREFIX}.meta.json"
if [[ "$VARIANT" == "packed" ]]; then
  echo "[DONE] graph_l1_seeds=${GRAPH_PREFIX}.l1_seeds.bin"
  echo "[DONE] owner_map=${GRAPH_PREFIX}.owner.bin"
  echo "[DONE] dist_map=${GRAPH_PREFIX}.dist.bin"
  echo "[DONE] ordered_clusters=${GRAPH_PREFIX}.ordered_clusters.json"
  echo "[DONE] write_order=${GRAPH_PREFIX}.write_order.bin"
  echo "[DONE] payload_sizes=${GRAPH_PREFIX}.payload_sizes.bin"
  echo "[DONE] pack_method=${PACK_METHOD} (dp_hybrid)"
  echo "[DONE] target_block_bytes=${TARGET_BLOCK_BYTES} (0 means device logic block size)"
fi
echo "[DONE] artifacts_md=$ARTIFACT_MD"
echo "[DONE] graph_mode=${GRAPH_MODE} (auto|rebuild|reuse|resume)"
echo "[DONE] dump_json=$DUMP_JSON"
if [[ -n "$REC_JSON" ]]; then
  echo "[DONE] recommended_buckets=$REC_JSON"
fi
