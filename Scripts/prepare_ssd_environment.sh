#!/bin/bash
# ============================================================================
# SSD环境准备脚本 - 只准备环境，不运行测试
# ============================================================================
# 
# 功能：
# 1. 卸载磁盘挂载点
# 2. 清理分区表
# 3. 清空FTL
# 4. FTL预热
# 5. 不运行测试
#
# 用法：
#   ./Scripts/prepare_ssd_environment.sh [--disk DISK] [--cool-time SEC] [-y|--yes]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "脚本目录: $SCRIPT_DIR"
source "${SCRIPT_DIR}/common.sh"
set_log_prefix "SSD_PREP"

DISK="${DISK:-/dev/nvme0n1}"
COOL_TIME="${COOL_TIME:-30}"
ASSUME_YES="${ASSUME_YES:-0}"

# ================================
# 主流程
# ================================
main() {
    log "===== SSD 环境准备开始 ====="
    log "磁盘: $DISK"
    log "冷却时间: ${COOL_TIME}秒"
    
    if ! prepare_ssd_environment_full "$DISK" "$COOL_TIME" "$ASSUME_YES"; then
        err "SSD 环境准备失败"
        exit 1
    fi

    log "现在可以运行实验了"
}

# 支持命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case $1 in
        --disk)
            DISK="$2"
            shift 2
            ;;
        --cool-time)
            COOL_TIME="$2"
            shift 2
            ;;
        -y|--yes)
            ASSUME_YES=1
            shift
            ;;
        *)
            err "未知参数: $1"
            echo "用法: $0 [--disk DISK] [--cool-time SEC] [-y|--yes]"
            exit 1
            ;;
    esac
done

main
