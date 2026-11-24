#!/bin/bash
# ============================================================================
# 公共函数库 - 供其他脚本使用
# ============================================================================
# 
# 用法：在脚本开头添加：
#   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#   WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
#   source "${WORKSPACE_ROOT}/labScripts/common.sh"
# ============================================================================

# ============================================================================
# 颜色和日志函数
# ============================================================================
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数（带前缀）
# 用法：log_with_prefix "PREFIX" "message"
log_with_prefix() {
    local prefix="$1"
    shift
    echo -e "${GREEN}[${prefix}]${NC} $*"
}

warn_with_prefix() {
    local prefix="$1"
    shift
    echo -e "${YELLOW}[${prefix}]${NC} $*"
}

err_with_prefix() {
    local prefix="$1"
    shift
    echo -e "${RED}[${prefix}]${NC} $*" >&2
}

info_with_prefix() {
    local prefix="$1"
    shift
    echo -e "${BLUE}[${prefix}]${NC} $*"
}

# 设置日志前缀（供脚本调用）
set_log_prefix() {
    export LOG_PREFIX="${1:-SCRIPT}"
}

# 带前缀的日志函数（使用环境变量 LOG_PREFIX）
log() { log_with_prefix "${LOG_PREFIX:-SCRIPT}" "$@"; }
warn() { warn_with_prefix "${LOG_PREFIX:-SCRIPT}" "$@"; }
err() { err_with_prefix "${LOG_PREFIX:-SCRIPT}" "$@"; }
info() { info_with_prefix "${LOG_PREFIX:-SCRIPT}" "$@"; }

# ============================================================================
# 系统缓存清理
# ============================================================================
clear_cache() {
    log "清理系统缓存"
    sudo sync 2>/dev/null || true
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
}

# ============================================================================
# 进程管理
# ============================================================================
wait_for_exit() {
    local pattern="$1"
    local label="$2"
    local timeout="${3:-300}"
    local waited=0

    while pgrep -f "$pattern" >/dev/null 2>&1; do
        if (( waited == 0 )); then
            log "等待 ${label} 退出..."
        fi
        sleep 1
        waited=$(( waited + 1 ))
        if (( waited >= timeout )); then
            warn "${label} 超过 ${timeout}s 仍在运行，尝试强制结束"
            pkill -9 -f "$pattern" || true
            break
        fi
    done
}

ensure_clean_environment() {
    local exec_patterns=("$@")
    
    for pattern in "${exec_patterns[@]}"; do
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            warn "检测到残留进程: $pattern"
            pkill -9 -f "$pattern" || true
        fi
        wait_for_exit "$pattern" "$(basename $pattern)" 30
    done
}

# ============================================================================
# SSD 环境准备函数
# ============================================================================

# 识别磁盘状态
detect_ssd_state() {
    local disk="$1"
    local disk_sys="/sys/block/$(basename $disk)"
    
    log "检测磁盘当前状态..."

    # 检查整盘是否挂载
    local whole_mount=$(mount | grep "^$disk " || true)
    local whole_mount_point=""
    if [ -n "$whole_mount" ]; then
        log "发现整盘被挂载到: $(echo $whole_mount | awk '{print $3}')"
        whole_mount_point=$(echo $whole_mount | awk '{print $3}')
    fi

    # 检查磁盘是否被格式化为文件系统
    local fs_type=$(lsblk -no FSTYPE $disk 2>/dev/null || echo "")
    if [ -n "$fs_type" ]; then
        log "整盘包含文件系统: $fs_type"
    fi

    # 检查是否存在分区
    local parts=$(lsblk -ln -o NAME $disk 2>/dev/null | tail -n +2 || echo "")
    if [ -n "$parts" ]; then
        log "发现分区：$(echo $parts)"
    fi
    
    # 返回挂载点（通过全局变量）
    export SSD_WHOLE_MOUNT_POINT="$whole_mount_point"
}

# 卸载所有挂载点
unmount_ssd_all() {
    local disk="$1"
    
    log "卸载磁盘的全部挂载点..."

    # 卸载整盘挂载
    if [ -n "$SSD_WHOLE_MOUNT_POINT" ]; then
        log "卸载整盘挂载: $SSD_WHOLE_MOUNT_POINT"
        sudo umount -l "$SSD_WHOLE_MOUNT_POINT" || true
    fi

    # 卸载分区挂载
    for p in $(lsblk -ln -o NAME $disk 2>/dev/null | tail -n +2 || echo ""); do
        if mount | grep -q "/dev/$p"; then
            log "卸载 /dev/$p ..."
            sudo umount -l "/dev/$p" || true
        fi
    done
}

# 清理分区表
cleanup_ssd_partition_table() {
    local disk="$1"
    local disk_sys="/sys/block/$(basename $disk)"
    
    log "清空磁盘 GPT/MBR 分区表..."

    sudo wipefs -a $disk || true
    sudo sgdisk --zap-all $disk || true

    log "尝试让内核刷新分区表..."
    sudo partprobe $disk || true

    log "使用 hdparm -z 强制刷新内核分区表缓存..."
    sudo hdparm -z $disk || true
    
    if [ -w "$disk_sys/device/rescan" ]; then
        echo 1 | sudo tee "$disk_sys/device/rescan" >/dev/null 2>&1 || true
    fi
}

# 强制 detach + PCI rescan
force_ssd_detach_rescan() {
    local disk="$1"
    local disk_sys="/sys/block/$(basename $disk)"
    
    log "执行 NVMe detach + PCI rescan..."

    if [ -e "$disk_sys/device/delete" ]; then
        echo 1 | sudo tee "$disk_sys/device/delete" >/dev/null || true
        sleep 1
    fi

    echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null
    sleep 1
}

# 清空 FTL
reset_ssd_ftl() {
    local disk="$1"
    
    log "尝试 blkdiscard 清空 FTL..."

    if sudo blkdiscard $disk 2>/dev/null; then
        log "blkdiscard 成功"
        return
    fi

    warn "blkdiscard 失败，尝试 nvme format..."
    if sudo nvme format $disk -s 1 -y 2>/dev/null; then
        log "nvme format 成功"
        return
    fi

    err "format failed，执行 detach + retry"
    force_ssd_detach_rescan "$disk"

    sudo blkdiscard $disk || sudo nvme format $disk -s 1 -y
}

# 检查 SSD 识别状态
check_ssd_recognized() {
    local disk="$1"
    local disk_name=$(basename $disk)
    
    log "检查 NVMe 是否被识别为 SSD..."

    if [ -f "/sys/block/$disk_name/queue/rotational" ]; then
        if [ "$(cat /sys/block/$disk_name/queue/rotational)" = "0" ]; then
            log "OK: rotational = 0，识别为 SSD"
        else
            err "错误：rotational=1，系统误识别为 HDD！"
            return 1
        fi
    else
        warn "无法读取 rotational 信息，跳过检查"
    fi

    if [ -f "/sys/block/$disk_name/queue/scheduler" ]; then
        local sched=$(cat /sys/block/$disk_name/queue/scheduler)
        log "I/O scheduler: $sched"
    fi
}

# FTL warmup
warmup_ssd_ftl() {
    local disk="$1"
    local cool_time="${2:-30}"
    
    log "FTL 预热（顺序写入 1GB）..."
    sudo dd if=/dev/zero of=$disk bs=4M count=256 oflag=direct status=none

    log "等待 SSD 控制器冷却 ${cool_time}s..."
    sleep $cool_time
}

# 完整的 SSD 环境准备流程
prepare_ssd_environment_full() {
    local disk="$1"
    local cool_time="${2:-30}"
    local assume_yes="${3:-${ASSUME_YES:-0}}"
    
    log "===== SSD 环境准备开始 ====="
    log "磁盘: $disk"
    log "冷却时间: ${cool_time}秒"
    
    warn "警告: 这将清空磁盘 $disk 的所有数据！"
    if [[ "$assume_yes" == "1" || "$assume_yes" == "yes" || "$assume_yes" == "YES" ]]; then
        log "已启用 -y/--yes，跳过交互确认"
    else
        read -p "确认继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            err "用户取消操作"
            return 1
        fi
    fi

    detect_ssd_state "$disk"
    unmount_ssd_all "$disk"
    cleanup_ssd_partition_table "$disk"
    reset_ssd_ftl "$disk"
    check_ssd_recognized "$disk" || return 1
    warmup_ssd_ftl "$disk" "$cool_time"

    log "===== SSD 环境准备完成 ====="
    return 0
}
