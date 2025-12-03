#!/bin/bash
# 切换终端地址，否则会无法连接外网
# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="http://127.0.0.1:7890"
# export all_proxy="socks5://127.0.0.1:7891"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
URL_FILE="$DATA_DIR/enwiki-20240420-extracted_urls.txt.gz"
SAMPLED_URLS="$DATA_DIR/subsampled_positive_urls.txt"
OUTPUT_WARC="$DATA_DIR/positive_samples.warc"

# 1. 检查源文件是否存在
if [ ! -f "$URL_FILE" ]; then
    echo "❌ 错误: 找不到源文件 $URL_FILE"
    exit 1
fi

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "URL_FILE: $URL_FILE"

echo "1. 正在抽取 URL (使用 gunzip -c)..."
gunzip -c "$URL_FILE" | gshuf -n 5000 > "$SAMPLED_URLS"

# 检查抽取是否成功
if [ ! -s "$SAMPLED_URLS" ]; then
    echo "❌ 错误: URL列表生成失败, 文件为空。"
    exit 1
fi

echo "Done! 5000 URLs saved to $SAMPLED_URLS"

echo "2. 开始爬取网页 (Wget)... 这可能需要几分钟"
# 删除旧的空 WARC 文件，防止 wget 续传或报错
rm -f "${OUTPUT_WARC}.gz"

# 使用 wget 的 WARC 模式
wget --timeout=5 \
     --tries=1 \
     -i "$SAMPLED_URLS" \
     --warc-file="${OUTPUT_WARC%.warc}" \
     -O /dev/null \
     --no-check-certificate

echo "完成！正样本保存在: $OUTPUT_WARC.gz"