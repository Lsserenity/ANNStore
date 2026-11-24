#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAB_SCRIPTS_DIR="${ROOT_DIR}/Scripts"

# 引入公共函数
source "${LAB_SCRIPTS_DIR}/common.sh"
set_log_prefix "FS_SETUP"

# 默认参数（可被环境变量或命令行覆盖）
DISK="${DISK:-/dev/nvme0n1}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/nvme0n1}"
NUM_FILES="${NUM_FILES:-10000000}"    # basefile 默认 10M
FILE_SIZE="${FILE_SIZE:-4096}"
THREADS="${THREADS:-64}"
FS_TYPE="${FS_TYPE:-ext4}"
LAYOUT="multi"                        # single / multi
TREE_FANOUT="${TREE_FANOUT:-100}"     # 仅用于估算/透传 fanout
TREE_ROOT=""                         # 可选：覆盖 tree 根目录
PATTERN=""                           # 可选：zero | z | random | random-per-file
QUIET=0
REQUIRED_INODES=0
SKIP_LAB1=0
RESUME_MODE=0

# 默认 C++ 可执行文件（可通过环境变量 FS_PREPARE_BIN 覆盖）
# Use the locally built artifact executable unless explicitly overridden.
DEFAULT_PREPARE_BIN_LOCAL="${ROOT_DIR}/execs/FsPrepareFilesTrace"
if [[ -n "${FS_PREPARE_BIN:-}" ]]; then
    PREPARE_BIN="${FS_PREPARE_BIN}"
elif [[ -x "${DEFAULT_PREPARE_BIN_LOCAL}" ]]; then
    PREPARE_BIN="${DEFAULT_PREPARE_BIN_LOCAL}"
else
    PREPARE_BIN="${DEFAULT_PREPARE_BIN_LOCAL}"
fi

usage() {
    cat <<EOF
用法: sudo bash Scripts/fs-workspace/setup_fs_workspace.sh [选项]

选项:
  --disk PATH           裸盘设备路径 (默认: ${DISK})
  --mount-point PATH    挂载点目录 (默认: ${MOUNT_POINT})
  --num-files N         生成文件总数 (默认: ${NUM_FILES}, 建议=10000000)
  --file-size BYTES     每个文件大小 (默认: ${FILE_SIZE})
  --threads N           生成文件使用的线程数 (默认: ${THREADS})
  --fs-type TYPE        文件系统类型: ext4 | xfs | btrfs | ext3 | ext2 | vfat (默认: ${FS_TYPE})
  --layout single       只生成单层结构 (baselineA, lab1)
  --layout multi        只生成多层树结构 (baselineB, tree, 默认)
  --tree-fanout N       tree 二层 fanout（默认: ${TREE_FANOUT}）
  --tree-root PATH      tree 根目录（默认: <mount-point>/tree）
  --pattern TYPE        文件内容模式: zero | z | random | random-per-file（默认: z）
  --required-inodes N   强制指定 mkfs inode 数量（覆盖自动估算）
  --resume              已格式化/挂载，跳过 mkfs，直接续写
  --skip-lab1           仅对 multi 布局生效，跳过 lab1
  --prepare-bin PATH    指定 prepare_files 可执行文件路径
  --quiet, -q           静默模式，prepare_files 输出重定向到日志
  --help, -h            显示本帮助

示例:
  # baselineA：单层结构 + 10M 文件
  sudo bash Scripts/fs-workspace/setup_fs_workspace.sh --layout single --num-files 10000000

  # baselineB：多层结构 + 10M 文件
  sudo bash Scripts/fs-workspace/setup_fs_workspace.sh --layout multi --num-files 10000000
EOF
}

# ----------------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --disk)
            DISK="$2"; shift 2;;
        --mount-point)
            MOUNT_POINT="$2"; shift 2;;
        --num-files)
            NUM_FILES="$2"; shift 2;;
        --file-size)
            FILE_SIZE="$2"; shift 2;;
        --threads)
            THREADS="$2"; shift 2;;
        --fs-type)
            FS_TYPE="$2"; shift 2;;
        --layout)
            LAYOUT="$2"; shift 2;;
        --tree-fanout)
            TREE_FANOUT="$2"; shift 2;;
        --tree-root)
            TREE_ROOT="$2"; shift 2;;
        --pattern)
            PATTERN="$2"; shift 2;;
        --required-inodes)
            REQUIRED_INODES="$2"; shift 2;;
        --prepare-bin)
            PREPARE_BIN="$2"; shift 2;;
        --resume)
            RESUME_MODE=1; shift 1;;
        --skip-lab1)
            SKIP_LAB1=1; shift 1;;
        --quiet|-q)
            QUIET=1; shift 1;;
        --help|-h)
            usage; exit 0;;
        *)
            warn "忽略未知参数: $1"
            shift 1;;
    esac
done

# ----------------------------------------------------------------------------
# 基础检查
# ----------------------------------------------------------------------------
check_root() {
    if [[ $EUID -ne 0 ]]; then
        err "请用 root 权限运行本脚本（涉及 mkfs/mount）"
        exit 1
    fi
}

check_disk() {
    if [[ ! -b "$DISK" ]]; then
        err "块设备不存在: $DISK"
        exit 1
    fi
    log "检测到磁盘: $DISK"
}

ensure_mkfs_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        err "未找到可执行文件: ${tool}"
        case "$tool" in
            mkfs.btrfs)
                err "请安装: btrfs-progs (Debian/Ubuntu: sudo apt install btrfs-progs)"
                ;;
            mkfs.xfs)
                err "请安装: xfsprogs (Debian/Ubuntu: sudo apt install xfsprogs)"
                ;;
            mkfs.ext2|mkfs.ext3|mkfs.ext4)
                err "请安装: e2fsprogs (Debian/Ubuntu: sudo apt install e2fsprogs)"
                ;;
            mkfs.vfat)
                err "请安装: dosfstools (Debian/Ubuntu: sudo apt install dosfstools)"
                ;;
        esac
        exit 1
    fi
}

ensure_kernel_fs_support() {
    local fs="$1"
    if grep -qw "$fs" /proc/filesystems; then
        return 0
    fi

    if command -v modprobe >/dev/null 2>&1; then
        if ! modprobe "$fs" >/dev/null 2>&1; then
            err "内核模块加载失败: ${fs}"
            err "请确认内核已启用该文件系统模块"
            exit 1
        fi
        if ! grep -qw "$fs" /proc/filesystems; then
            err "已尝试 modprobe ${fs}，但 /proc/filesystems 中仍未出现该类型"
            err "请确认内核支持并已安装对应模块"
            exit 1
        fi
    else
        err "未找到 modprobe，无法自动加载 ${fs} 内核模块"
        err "请确认内核已启用 ${fs} 文件系统支持"
        exit 1
    fi
}

ensure_mount_ready_for_resume() {
    if ! mountpoint -q "$MOUNT_POINT"; then
        err "resume 模式要求挂载点 $MOUNT_POINT 已挂载"
        exit 1
    fi

    local src
    src=$(findmnt -n -o SOURCE --target "$MOUNT_POINT" 2>/dev/null || "")
    if [[ -n "$src" ]]; then
        log "挂载点来源: $src"
        if [[ -n "$DISK" && "$src" != "$DISK" ]]; then
            warn "挂载点来源 ($src) 与指定磁盘 ($DISK) 不一致，仍继续执行"
        fi
    fi

    local fstype
    fstype=$(findmnt -n -o FSTYPE --target "$MOUNT_POINT" 2>/dev/null || "")
    if [[ -n "$FS_TYPE" && "$fstype" != "$FS_TYPE" ]]; then
        warn "resume 模式下挂载点文件系统类型为 '$fstype'（预期 ${FS_TYPE}）"
    fi

    log "resume 模式：跳过 mkfs/mount，直接在现有挂载点生成文件"
}

estimate_tree_directory_count() {
    local fanout="$1"
    local f1="$fanout"
    local f2=$((fanout * fanout))
    # tree 根节点 + 第一层 + 第二层
    echo $((1 + f1 + f2))
}

calculate_required_inodes() {
    local need_lab1="$1"
    local need_tree="$2"
    local num_files="$3"
    local fanout="$4"

    local total=0
    if [[ "$need_lab1" -eq 1 ]]; then
        total=$((total + num_files + 1)) # lab1 目录
    fi
    if [[ "$need_tree" -eq 1 ]]; then
        local dirs
        dirs=$(estimate_tree_directory_count "$fanout")
        total=$((total + num_files + dirs))
    fi

    if [[ "$total" -lt 0 ]]; then
        total=0
    fi

    local safety=$(( (total + 19) / 20 )) # 预留 5% 冗余
    total=$((total + safety + 100000))

    echo "$total"
}

format_and_mount() {
    log "检查挂载点: $MOUNT_POINT"

    # 如已挂载，先卸载
    if mountpoint -q "$MOUNT_POINT"; then
        log "挂载点已挂载，先卸载..."
        umount "$MOUNT_POINT" || true
    fi

    # 如整盘挂载在其他位置，也一并卸载
    if mount | grep -q "^$DISK "; then
        log "发现磁盘在其他位置挂载，尝试卸载..."
        umount "$DISK" || true
    fi

    log "将对 $DISK 执行 mkfs.${FS_TYPE} 并挂载到 $MOUNT_POINT"
    mkdir -p "$MOUNT_POINT"

    local mkfs_args=()
    local mkfs_cmd=""
    case "$FS_TYPE" in
        ext4|ext3|ext2)
            mkfs_cmd="mkfs.${FS_TYPE}"
            mkfs_args+=(-F)
            if [[ "${REQUIRED_INODES:-0}" -gt 0 ]]; then
                mkfs_args+=(-N "$REQUIRED_INODES")
            fi
            if [[ $QUIET -eq 1 ]]; then
                mkfs_args+=(-q)
            fi
            ;;
        xfs)
            mkfs_cmd="mkfs.xfs"
            mkfs_args+=(-f)
            if [[ "${REQUIRED_INODES:-0}" -gt 0 ]]; then
                warn "xfs 不支持 mkfs 的 inode 预留参数，已忽略 --required-inodes"
            fi
            if [[ $QUIET -eq 1 ]]; then
                mkfs_args+=(-q)
            fi
            ;;
        btrfs)
            mkfs_cmd="mkfs.btrfs"
            mkfs_args+=(-f)
            if [[ "${REQUIRED_INODES:-0}" -gt 0 ]]; then
                warn "btrfs 不支持 mkfs 的 inode 预留参数，已忽略 --required-inodes"
            fi
            if [[ $QUIET -eq 1 ]]; then
                mkfs_args+=(-q)
            fi
            ;;
        vfat)
            mkfs_cmd="mkfs.vfat"
            mkfs_args+=(-I)
            if [[ "${REQUIRED_INODES:-0}" -gt 0 ]]; then
                warn "vfat 不支持 mkfs 的 inode 预留参数，已忽略 --required-inodes"
            fi
            ;;
        *)
            err "不支持的文件系统类型: ${FS_TYPE} (可选: ext4/xfs/btrfs/ext3/ext2/vfat)"
            exit 1
            ;;
    esac

    log "执行 ${mkfs_cmd} ${mkfs_args[*]} $DISK"
    "$mkfs_cmd" "${mkfs_args[@]}" "$DISK"

    mount -t "$FS_TYPE" "$DISK" "$MOUNT_POINT"
    log "格式化并挂载完成: $DISK -> $MOUNT_POINT"
}

run_prepare_files() {
    if [[ ! -x "$PREPARE_BIN" ]]; then
        err "prepare_files 可执行文件不存在或不可执行: $PREPARE_BIN"
        err "请先编译该可执行文件，或通过 --prepare-bin/FS_PREPARE_BIN 指定路径"
        exit 1
    fi

    local extra_args=()
    case "$LAYOUT" in
        single)
            # 对应 new_setup_fs.sh 中的 --no-tree
            extra_args+=(--no-tree)
            log "选择布局: single（baselineA，单层结构，目录一般为 ${MOUNT_POINT}/lab1）"
            if [[ -n "$PATTERN" ]]; then
                extra_args+=(--pattern "$PATTERN")
                log "文件内容模式: ${PATTERN}"
            fi
            ;;
        multi)
            log "选择布局: multi（baselineB/benchmark tree）"
            if [[ $SKIP_LAB1 -eq 1 ]]; then
                extra_args+=(--skip-lab1)
                log "跳过 lab1 生成 (--skip-lab1)，仅构建 tree 结构"
            fi
            extra_args+=(--fanout "$TREE_FANOUT")
            if [[ -n "$TREE_ROOT" ]]; then
                extra_args+=(--tree-root "$TREE_ROOT")
                log "tree 根目录覆盖: ${TREE_ROOT}"
            fi
            if [[ -n "$PATTERN" ]]; then
                extra_args+=(--pattern "$PATTERN")
                log "文件内容模式: ${PATTERN}"
            fi
            ;;
        *)
            warn "未知布局: $LAYOUT，默认使用 multi"
            LAYOUT="multi"
            ;;
    esac

    log "开始运行多线程文件生成程序"
    log "  mount-point = $MOUNT_POINT"
    log "  num-files   = $NUM_FILES"
    log "  file-size   = $FILE_SIZE"
    log "  threads     = $THREADS"

    local status=0
    if [[ $QUIET -eq 1 ]]; then
        local log_dir="${ROOT_DIR}/logs"
        mkdir -p "$log_dir"
        local layout_tag
        if [[ "$LAYOUT" == "single" ]]; then
            layout_tag="single"
        else
            layout_tag="multi"
        fi
        local log_file="${log_dir}/prepare_fs_${layout_tag}_$(date +%Y%m%d_%H%M%S).log"
        log "静默模式：prepare_files 输出重定向到 ${log_file}"

        set +e
        "$PREPARE_BIN" \
            --mount-point "$MOUNT_POINT" \
            --num-files "$NUM_FILES" \
            --file-size "$FILE_SIZE" \
            --threads "$THREADS" \
            "${extra_args[@]}" \
            >"$log_file" 2>&1
        status=$?
        set -e
    else
        set +e
        "$PREPARE_BIN" \
            --mount-point "$MOUNT_POINT" \
            --num-files "$NUM_FILES" \
            --file-size "$FILE_SIZE" \
            --threads "$THREADS" \
            "${extra_args[@]}"
        status=$?
        set -e
    fi

    if [[ $status -ne 0 ]]; then
        err "prepare_files 执行失败，退出码: $status"
        exit "$status"
    fi

    log "prepare_files 执行完成"
    log "注意：单层结构通常位于 ${MOUNT_POINT}/lab1，多层结构位于 ${MOUNT_POINT}/tree"
}

main() {
    log "===== 文件系统工作空间准备开始 ====="
    log "磁盘:        $DISK"
    log "挂载点:      $MOUNT_POINT"
    log "文件数量:    $NUM_FILES"
    log "文件大小:    $FILE_SIZE"
    log "线程数:      $THREADS"
    log "文件系统:    $FS_TYPE"
    log "prepare_bin: $PREPARE_BIN"

    local need_lab1=1
    local need_tree=0
    if [[ "$LAYOUT" == "multi" ]]; then
        need_tree=1
        if [[ $SKIP_LAB1 -eq 1 ]]; then
            need_lab1=0
        fi
    fi

    if [[ "${REQUIRED_INODES:-0}" -le 0 ]]; then
        REQUIRED_INODES=$(calculate_required_inodes "$need_lab1" "$need_tree" "$NUM_FILES" "$TREE_FANOUT")
    fi
    log "预估 inode 需求: ${REQUIRED_INODES} (lab1=${need_lab1}, tree=${need_tree}, fanout=${TREE_FANOUT})"

    if [[ $RESUME_MODE -eq 1 ]]; then
        ensure_mount_ready_for_resume
    else
        check_root
        check_disk
        case "$FS_TYPE" in
            ext4|ext3|ext2)
                ensure_mkfs_tool "mkfs.${FS_TYPE}"
                ;;
            xfs)
                ensure_mkfs_tool "mkfs.xfs"
                ensure_kernel_fs_support "xfs"
                ;;
            btrfs)
                ensure_mkfs_tool "mkfs.btrfs"
                ensure_kernel_fs_support "btrfs"
                ;;
            vfat)
                ensure_mkfs_tool "mkfs.vfat"
                ;;
            *)
                err "不支持的文件系统类型: ${FS_TYPE} (可选: ext4/xfs/btrfs/ext3/ext2/vfat)"
                exit 1
                ;;
        esac
        format_and_mount
    fi
    run_prepare_files

    log "===== 文件系统工作空间准备完成 ====="
    log "可用于后续 baselineA/baselineB 实验"
}

main "$@"
