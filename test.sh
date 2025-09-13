#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm"

# 数据路径（你本地）
TEST_DATA_DIR="/media/ps/PS10T/changkai/MICCAI2025ToothFairtTask1/test"
IMAGES_DIR="$TEST_DATA_DIR/imagesTr"
ALG_OUTPUT_DIR="$TEST_DATA_DIR/algorithm-output"


# 清理 & 准备 staging input
rm -rf "$SCRIPTPATH/test/input" "$SCRIPTPATH/test/output"
mkdir -p "$SCRIPTPATH/test/input/images/cbct"
mkdir -p "$SCRIPTPATH/test/output/images/oral-pharyngeal-segmentation"
mkdir -p "$SCRIPTPATH/test/output/metadata"
mkdir -p "$ALG_OUTPUT_DIR"

# 复制并重命名测试图像
echo "Copying test images from $IMAGES_DIR..."
counter=1
for file in "$IMAGES_DIR"/*.nii.gz; do
    if [ -f "$file" ]; then
        new_name=$(printf "%03d_0000.nii.gz" $counter)
        cp "$file" "$SCRIPTPATH/test/input/images/cbct/$new_name"
        echo "  $(basename "$file") -> $new_name"
        counter=$((counter + 1))
    fi
done

echo "Running Docker container..."
docker run --rm \
    --gpus=all \
    --memory=30g \
    --memory-swap=30g \
    --network="none" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    --shm-size="128m" \
    --pids-limit="256" \
    -v "$SCRIPTPATH/test/input/images/cbct":/input/images/cbct:ro \
    -v "$SCRIPTPATH/test/output":/output/ \
    "$DOCKER_TAG" \
    --input_path /input/images/cbct \
    --output_path /output/images/oral-pharyngeal-segmentation \
    --metadata_path /output/metadata

# 结果本来就在 host 的 test/output 里，搬一份到统一输出
echo "Copying results to final output dir: $ALG_OUTPUT_DIR"
rm -rf "$ALG_OUTPUT_DIR"/*
cp -r "$SCRIPTPATH/test/output/"* "$ALG_OUTPUT_DIR/"

echo "Done. Outputs are in: $ALG_OUTPUT_DIR"
