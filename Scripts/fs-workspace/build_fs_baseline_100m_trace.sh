#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAB_SCRIPTS_DIR="${ROOT_DIR}/Scripts"
source "${LAB_SCRIPTS_DIR}/common.sh"
set_log_prefix "FS_100M_TRACE"

DEFAULT_DISK="/dev/nvme0n1"
DEFAULT_MOUNT_POINT="/mnt/nvme0n1"
DEFAULT_TRACE_PATH="${ROOT_DIR}/DownLoadDataSets/workload_sizes_100M.npy"

DISK="$DEFAULT_DISK"
MOUNT_POINT="$DEFAULT_MOUNT_POINT"
ROUNDS=100
START_INDEX=0
NUM_FILES=1000000
THREADS=64
FS_TYPE="ext4"
TREE_FANOUT=100
PATTERN=""
QUIET=0
PREPARE_BIN=""
RESUME_FIRST=0
TRACE_PATH="$DEFAULT_TRACE_PATH"
TRACE_OFFSET=-1

usage() {
    cat <<EOF2
用法: sudo bash Scripts/fs-workspace/build_fs_baseline_100m_trace.sh [选项]

选项:
  --disk PATH           裸盘设备路径 (默认: ${DISK})
  --mount-point PATH    挂载点目录 (默认: ${MOUNT_POINT})
  --rounds N            轮数 (默认: ${ROUNDS})
  --start-index N       起始轮次下标 (默认: ${START_INDEX})
  --num-files N         每轮生成文件数 (默认: ${NUM_FILES})
  --threads N           线程数 (默认: ${THREADS})
  --fs-type TYPE        文件系统类型: ext4 | xfs | btrfs | ext3 | ext2 | vfat (默认: ${FS_TYPE})
  --tree-fanout N       tree 二层 fanout (默认: ${TREE_FANOUT})
  --pattern TYPE        文件内容模式: zero | z | random | random-per-file (默认: z)
  --prepare-bin PATH    指定 FsPrepareFilesTrace 可执行文件
  --trace-path PATH     .npy 大小列表路径 (默认: ${TRACE_PATH})
  --trace-offset N      trace 起始偏移 (默认: start-index * num-files)
  --quiet, -q           静默模式
  --resume              首轮跳过 mkfs/mount，直接续写
  --help, -h            显示帮助

说明:
  每轮生成目录为:
    <mount-point>/tree/l{轮次}
  轮次内文件结构为:
    lX/lY/f_0.bin ~ f_99.bin (fanout=100 时)

  每文件大小取值规则:
    global_file_index = (当前轮次序号 * num_files) + 轮次内文件序号
    trace_index       = trace_offset + global_file_index
    file_size         = workload_sizes[trace_index]
EOF2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --disk)
            DISK="$2"; shift 2;;
        --mount-point)
            MOUNT_POINT="$2"; shift 2;;
        --rounds)
            ROUNDS="$2"; shift 2;;
        --start-index)
            START_INDEX="$2"; shift 2;;
        --num-files)
            NUM_FILES="$2"; shift 2;;
        --threads)
            THREADS="$2"; shift 2;;
        --fs-type)
            FS_TYPE="$2"; shift 2;;
        --tree-fanout)
            TREE_FANOUT="$2"; shift 2;;
        --pattern)
            PATTERN="$2"; shift 2;;
        --prepare-bin)
            PREPARE_BIN="$2"; shift 2;;
        --trace-path)
            TRACE_PATH="$2"; shift 2;;
        --trace-offset)
            TRACE_OFFSET="$2"; shift 2;;
        --quiet|-q)
            QUIET=1; shift 1;;
        --resume)
            RESUME_FIRST=1; shift 1;;
        --help|-h)
            usage; exit 0;;
        *)
            warn "忽略未知参数: $1"
            shift 1;;
    esac
done

if [[ "$ROUNDS" -le 0 || "$NUM_FILES" -le 0 ]]; then
    err "ROUNDS 和 NUM_FILES 必须大于 0"
    exit 1
fi

if [[ "$TRACE_OFFSET" -lt 0 ]]; then
    TRACE_OFFSET=$((START_INDEX * NUM_FILES))
fi

if [[ -z "$PREPARE_BIN" ]]; then
    PREPARE_BIN="${ROOT_DIR}/execs/FsPrepareFilesTrace"
fi

SETUP_SCRIPT="${SCRIPT_DIR}/setup_fs_workspace.sh"
if [[ ! -x "$SETUP_SCRIPT" ]]; then
    err "找不到或无法执行: $SETUP_SCRIPT"
    exit 1
fi

if [[ ! -f "$TRACE_PATH" ]]; then
    err "未找到 trace 文件: $TRACE_PATH"
    exit 1
fi

if [[ ! -x "$PREPARE_BIN" ]]; then
    err "找不到或无法执行 trace prepare 程序: $PREPARE_BIN"
    exit 1
fi

trace_start=$TRACE_OFFSET
trace_count=$((ROUNDS * NUM_FILES))
log "trace 需求区间: path=${TRACE_PATH}, start=${trace_start}, count=${trace_count}"

log "开始分批构建: rounds=${ROUNDS}, start_index=${START_INDEX}, per_round_files=${NUM_FILES}"
log "挂载点: ${MOUNT_POINT}"
log "trace 路径: ${TRACE_PATH}"
log "trace prepare 程序: ${PREPARE_BIN}"
total_start_sec=$(date +%s)
total_start_iso=$(date -Iseconds)

for ((i = 0; i < ROUNDS; ++i)); do
    r=$((START_INDEX + i))
    round_trace_offset=$((TRACE_OFFSET + i * NUM_FILES))
    tree_root="${MOUNT_POINT}/tree/l${r}"
    log "轮次 ${r}: tree_root=${tree_root}, trace_offset=${round_trace_offset}, files=${NUM_FILES}"

    if [[ $r -eq $START_INDEX && $RESUME_FIRST -eq 0 ]]; then
        total_files=$((ROUNDS * NUM_FILES))
        per_round_dirs=$((1 + TREE_FANOUT + TREE_FANOUT * TREE_FANOUT))
        total_dirs=$((1 + ROUNDS * per_round_dirs))
        required_inodes=$((total_files + total_dirs))
        safety=$(( (required_inodes + 19) / 20 ))
        required_inodes=$((required_inodes + safety + 100000))
        log "预计总 inode 需求: ${required_inodes} (total_files=${total_files}, total_dirs=${total_dirs})"
    fi

    round_start_sec=$(date +%s)
    round_start_iso=$(date -Iseconds)
    args=(
        --disk "$DISK"
        --mount-point "$MOUNT_POINT"
        --layout multi
        --skip-lab1
        --num-files "$NUM_FILES"
        --file-size 4096
        --threads "$THREADS"
        --fs-type "$FS_TYPE"
        --tree-fanout "$TREE_FANOUT"
        --tree-root "$tree_root"
        --prepare-bin "$PREPARE_BIN"
    )

    if [[ -n "$PATTERN" ]]; then
        args+=(--pattern "$PATTERN")
    fi
    if [[ $QUIET -eq 1 ]]; then
        args+=(--quiet)
    fi

    if [[ $r -eq $START_INDEX ]]; then
        if [[ $RESUME_FIRST -eq 1 ]]; then
            args+=(--resume)
        else
            args+=(--required-inodes "$required_inodes")
        fi
    else
        args+=(--resume)
    fi

    sudo FS_TRACE_PATH="$TRACE_PATH" \
        FS_TRACE_START="$round_trace_offset" \
        FS_TRACE_COUNT="$NUM_FILES" \
        bash "$SETUP_SCRIPT" "${args[@]}"
    round_end_sec=$(date +%s)
    round_end_iso=$(date -Iseconds)
    round_duration=$((round_end_sec - round_start_sec))
    log "轮次 ${r} 完成: start=${round_start_iso} end=${round_end_iso} duration=${round_duration}s"
done

total_end_sec=$(date +%s)
total_end_iso=$(date -Iseconds)
total_duration=$((total_end_sec - total_start_sec))
log "全部轮次完成: ${ROUNDS} 轮"
log "总耗时: start=${total_start_iso} end=${total_end_iso} duration=${total_duration}s"
