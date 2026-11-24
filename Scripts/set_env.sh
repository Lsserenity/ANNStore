#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
HNSWLIB_DIR="${EXTERNAL_DIR}/hnswlib"
SPDLOG_DIR="${EXTERNAL_DIR}/spdlog"
HNSWLIB_REF="${HNSWLIB_REF:-v0.8.0}"
JSON_VERSION="${JSON_VERSION:-v3.11.3}"

mkdir -p "${EXTERNAL_DIR}"

ensure_git_dependency() {
    local url="$1"
    local path="$2"

    if git -C "${path}" rev-parse --git-dir >/dev/null 2>&1; then
        echo "[SETUP] Reusing ${path}"
        return
    fi
    if [[ -e "${path}" ]] && [[ -n "$(ls -A "${path}" 2>/dev/null)" ]]; then
        echo "[ERROR] ${path} exists, is non-empty, and is not a Git checkout." >&2
        exit 1
    fi

    echo "[SETUP] Cloning ${url} -> ${path}"
    git clone "${url}" "${path}"
}

ensure_git_dependency "https://github.com/nmslib/hnswlib.git" "${HNSWLIB_DIR}"
ensure_git_dependency "https://github.com/gabime/spdlog.git" "${SPDLOG_DIR}"

# ANNStore's patch targets the hnswlib v0.8.0 internal class layout.
if grep -q "ANNFS_HNSW_MEMBER_CONTEXT" "${HNSWLIB_DIR}/hnswlib/hnswalg.h"; then
    echo "[SETUP] ANNStore hnswlib patch is already applied"
else
    rm -f "${HNSWLIB_DIR}/hnswlib.diff"
    if ! git -C "${HNSWLIB_DIR}" rev-parse --verify --quiet "${HNSWLIB_REF}^{commit}" >/dev/null; then
        echo "[SETUP] Fetching hnswlib ref ${HNSWLIB_REF}"
        git -C "${HNSWLIB_DIR}" fetch --tags origin
    fi
    echo "[SETUP] Checking out hnswlib ${HNSWLIB_REF}"
    git -C "${HNSWLIB_DIR}" checkout --quiet "${HNSWLIB_REF}"
    git -C "${HNSWLIB_DIR}" apply --check "${ROOT_DIR}/core/hnsw/hnswlib.diff"
    git -C "${HNSWLIB_DIR}" apply "${ROOT_DIR}/core/hnsw/hnswlib.diff"
    echo "[SETUP] Applied ANNStore hnswlib patch"
fi

JSON_PATH="${EXTERNAL_DIR}/json.hpp"
JSON_URL="https://raw.githubusercontent.com/nlohmann/json/${JSON_VERSION}/single_include/nlohmann/json.hpp"
if [[ ! -f "${JSON_PATH}" ]]; then
    echo "[SETUP] Downloading nlohmann/json ${JSON_VERSION}"
    if command -v wget >/dev/null 2>&1; then
        wget --timeout=30 -q "${JSON_URL}" -O "${JSON_PATH}"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --connect-timeout 30 -o "${JSON_PATH}" "${JSON_URL}"
    else
        echo "[ERROR] wget or curl is required to download json.hpp" >&2
        exit 1
    fi
else
    echo "[SETUP] Reusing ${JSON_PATH}"
fi

echo "[SETUP] Dependencies are ready"
