# ANNStore

ANNStore is an SSD-aware approximate nearest-neighbor (ANN) search prototype.
It optimizes HNSW search through an ANN-optimized storage engine and an I/O-aware ANN search engine.

## Repository structure

```text
core/        Core HNSW, I/O, configuration, and experiment code
LabTest/     Raw-device ANNStore experiment entry points
FSLabTest/   File-system baseline entry points
Scripts/     Dependency, SSD, layout, and baseline preparation scripts
```

Main executables produced in `execs/`:

| Executable | Description |
| --- | --- |
| `HnswOffsetIO` | *Direct Offset* baseline with serialized payload fetching |
| `HnswOffsetEarlyStop` | *Direct Offset* with topology-responsive *Iter-Management* |
| `HnswOffsetOverlapAdaptive` | *Direct Offset* with adaptive *Early-Fetch* |
| `HnswOffsetEarlyStopOverlapAdaptive` | Complete ANNStore with *Early-Fetch* and *Iter-Management* |
| `HnswMappedOffsetIO` | ANN search over the *Graph-guided Layout* |
| `HnswFsIO` / `HnswFsOpenOnly` | `FS-IO` / `OpenOnly` file-system baselines |
| `IndexBuild` | Parallel HNSW index construction |
| `HnswL1LayoutBase` / `HnswL1LayoutPacked` | *Graph-guided Layout* / *Fragmentation-aware Packing* builders |

## Requirements

- Linux with a C++17 compiler and CMake 3.22+
- A dedicated NVMe SSD for raw-device experiments
- Root access for SSD preparation and file-system experiments
- Sufficient memory and storage for the selected dataset and index

Ubuntu/Debian dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config git wget curl jq \
  libnuma-dev liburing-dev util-linux gdisk nvme-cli hdparm \
  e2fsprogs xfsprogs python3 python3-pip

python3 -m pip install --user numpy h5py faiss-cpu
```

Install third-party source dependencies:

```bash
bash ./Scripts/set_env.sh
```

This downloads `spdlog` and `nlohmann/json`, checks out `hnswlib v0.8.0`, and
applies `core/hnsw/hnswlib.diff`. The version is fixed because the ANNStore
HNSW extensions depend on the `v0.8.0` internal class layout.

## Build

```bash
bash ./build.sh Release
```

If the build directory contains a stale CMake cache:

```bash
bash ./build.sh Release --auto-clean
```

## Datasets and configuration

Copy [`configs/datasets.example.json`](configs/datasets.example.json) to
`configs/datasets.json` and update the machine-specific paths. Datasets and
prebuilt indexes are not included. Search executables use:

```text
<executable> <config.json> <dataset> [num_threads] [repeat_id] [options]
```

Update the example paths and match the distance metric to your vectors and saved
index. For raw payload retrieval, use the layout preparation script's generated
run configuration, which points to the relabeled index matching the disk layout.

## Prepare the index and storage layout

Build an HNSW index using the separate index-builder configuration:

```bash
./execs/IndexBuild configs/build-index.json deep100m float32
```

Prepare a base or packed ANNStore layout:

```bash
export ANNFS_TRACE_PATH=/data/annstore/workload_sizes.npy

sudo -E bash Scripts/ssd-layout/prepare_dataset_spanpool_front_lowmem.sh \
  deep100m packed configs/datasets.json
```

To condition a dedicated SSD before an experiment:

```bash
sudo bash Scripts/prepare_ssd_environment.sh \
  --disk /dev/nvme1n1 --cool-time 30
```

The last two commands may write directly to the selected device. Verify the
device path before running them.

## Run experiments

```bash
export ANNFS_END_RESULTS_ROOT="$PWD/json_results"
export ANNFS_END_RESULTS_RUN_ID="deep100m-vldb"

CONFIG=configs/datasets.json
DATASET=deep100m
THREADS=32
REPEAT=1

./execs/HnswOffsetIO "$CONFIG" "$DATASET" "$THREADS" "$REPEAT"
./execs/HnswOffsetEarlyStop "$CONFIG" "$DATASET" "$THREADS" "$REPEAT"
./execs/HnswOffsetOverlapAdaptive "$CONFIG" "$DATASET" "$THREADS" "$REPEAT"
./execs/HnswOffsetEarlyStopOverlapAdaptive \
  "$CONFIG" "$DATASET" "$THREADS" "$REPEAT"
```

Common overrides include `--search_ef`, `--stableiter`, `--stable_hops`.

### File-system baselines

The following command formats and mounts the selected device before generating
the baseline layout:

```bash
sudo bash Scripts/fs-workspace/setup_fs_workspace.sh \
  --disk /dev/nvme1n1 \
  --mount-point /mnt/annstore \
  --fs-type ext4 \
  --layout multi \
  --num-files 10000000 \
  --threads 64

./execs/HnswFsIO configs/datasets.json deep100m 32 1
./execs/HnswFsOpenOnly configs/datasets.json deep100m 32 1
```

## Results

Results are written under:

```text
json_results/<run-id>/<variant>/<dataset>/
  t<threads>/efS<search-ef>_qcnt<queries>_rpt_<repeat>/
```

For fair comparisons, use the same index, recall target, thread count, I/O
depth, query count, warm-up policy, and SSD conditioning policy. We recommend
at least three measured repetitions per configuration.

