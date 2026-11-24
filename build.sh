#!/bin/bash
set -euo pipefail

echo "=== Building FS_COMPARE Experiments ==="

usage() {
    cat <<'EOF'
Usage: ./build.sh [BuildType] [--auto-clean|--no-clean]

Examples:
  ./build.sh
  ./build.sh Release
  ./build.sh Debug --auto-clean

Options:
  --auto-clean  Cache mismatch 时自动清理 build/ 并重配
  --no-clean    Cache mismatch 时只提示并退出
  -h, --help    Show help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
BUILD_DIR="${ROOT_DIR}/build"

BUILD_TYPE="Release"
CLEAN_MODE="prompt" # prompt | always | never

while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto-clean)
            CLEAN_MODE="always"
            ;;
        --no-clean)
            CLEAN_MODE="never"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            BUILD_TYPE="$1"
            ;;
    esac
    shift
done

echo "Build Type: $BUILD_TYPE"

get_cache_value() {
    local key="$1"
    local cache_file="$2"
    sed -n "s#^${key}=##p" "$cache_file" | head -n 1
}

check_build_cache_mismatch() {
    local cache_file="$BUILD_DIR/CMakeCache.txt"
    local -n reasons_ref="$1"

    reasons_ref=()
    if [[ ! -f "$cache_file" ]]; then
        return 0
    fi

    local cache_src cache_gen cache_make_prog
    cache_src="$(get_cache_value "CMAKE_HOME_DIRECTORY:INTERNAL" "$cache_file")"
    cache_gen="$(get_cache_value "CMAKE_GENERATOR:INTERNAL" "$cache_file")"
    cache_make_prog="$(get_cache_value "CMAKE_MAKE_PROGRAM:FILEPATH" "$cache_file")"

    if [[ -n "$cache_src" && "$cache_src" != "$ROOT_DIR" ]]; then
        reasons_ref+=("source dir mismatch: cache='$cache_src', expected='$ROOT_DIR'")
    fi

    if [[ -n "$cache_gen" ]]; then
        if [[ "$cache_gen" == "Ninja" && ! -f "$BUILD_DIR/build.ninja" ]]; then
            reasons_ref+=("generator is Ninja but build.ninja is missing")
        fi
        if [[ "$cache_gen" == "Unix Makefiles" && ! -f "$BUILD_DIR/Makefile" ]]; then
            reasons_ref+=("generator is Unix Makefiles but Makefile is missing")
        fi
    fi

    if [[ -n "$cache_make_prog" && ! -e "$cache_make_prog" ]]; then
        reasons_ref+=("cached make program not found: $cache_make_prog")
    fi

    if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
        # CMake records the target subdirectory (for example build/core or
        # build/LabTest), not necessarily the build root itself.
        local dir_pattern="\"directory\": \"${BUILD_DIR}(/|\")"
        if command -v rg >/dev/null 2>&1; then
            if ! rg -q "$dir_pattern" "$BUILD_DIR/compile_commands.json"; then
                reasons_ref+=("compile_commands.json points to a different build directory")
            fi
        else
            if ! grep -qE "$dir_pattern" "$BUILD_DIR/compile_commands.json"; then
                reasons_ref+=("compile_commands.json points to a different build directory")
            fi
        fi
    fi
}

handle_cache_mismatch() {
    local reasons=("$@")
    if [[ ${#reasons[@]} -eq 0 ]]; then
        return 0
    fi

    echo "Detected inconsistent CMake cache in '$BUILD_DIR':"
    for reason in "${reasons[@]}"; do
        echo "  - $reason"
    done

    case "$CLEAN_MODE" in
        always)
            echo "Auto clean enabled: removing '$BUILD_DIR' ..."
            rm -rf "$BUILD_DIR"
            ;;
        never)
            echo "Abort: run './build.sh $BUILD_TYPE --auto-clean' to clean and rebuild."
            exit 1
            ;;
        *)
            if [[ -t 0 ]]; then
                read -r -p "Clean '$BUILD_DIR' and reconfigure now? [Y/n] " answer
                answer="${answer:-Y}"
                if [[ "$answer" =~ ^[Yy]$ ]]; then
                    rm -rf "$BUILD_DIR"
                else
                    echo "Abort: keep existing cache. Re-run with --auto-clean when needed."
                    exit 1
                fi
            else
                echo "Non-interactive shell detected. Re-run with --auto-clean to clean cache."
                exit 1
            fi
            ;;
    esac
}

declare -a mismatch_reasons=()
check_build_cache_mismatch mismatch_reasons
handle_cache_mismatch "${mismatch_reasons[@]}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# Remove binaries for retired targets. CMake does
# not delete outputs of targets that disappear from CMakeLists.txt.
cmake -E rm -f \
    "$ROOT_DIR/execs/HnswL1CentricFullWriter" \
    "$ROOT_DIR/execs/HnswOriginalPackedRelabel" \
    "$ROOT_DIR/execs/HnswRestoreOriginalLabels" \
    "$ROOT_DIR/execs/HnswL1LayoutPackedV0" \
    "$ROOT_DIR/execs/HnswL1LayoutPackedScore"
cmake -E rm -f \
    "$ROOT_DIR/execs/HnswOffsetMappedIORunner" \
    "$ROOT_DIR/execs/HnswGraphPersist"

# 编译所有实验，自动适配 Ninja / Unix Makefiles
cmake --build "$BUILD_DIR" -j"$(nproc)"

# 链接 compile_commands.json (为了代码补全)
SRC_COMPILE="$BUILD_DIR/compile_commands.json"
DST_COMPILE="$ROOT_DIR/compile_commands.json"
if [[ -f "$SRC_COMPILE" ]]; then
    ln -sf "$SRC_COMPILE" "$DST_COMPILE"
else
    echo "Warning: compile_commands.json not found at '$SRC_COMPILE'."
fi

echo "=== Build Done ==="
echo "Executables are located in: ./execs/"
ls -lh "$ROOT_DIR/execs/"
