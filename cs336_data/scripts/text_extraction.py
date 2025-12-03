# text_extraction_fixed.py
import os
import gzip
import sys
from fastwarc.warc import ArchiveIterator
from preprocessing import extract_text_from_html_bytes

warc_path = "data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

print("检查文件：", os.path.abspath(warc_path))
if not os.path.exists(warc_path):
    print("路径不存在，退出。")
    sys.exit(1)

max_records = 200
found_response = False

with gzip.open(warc_path, 'rb') as stream:
    for i, record in enumerate(ArchiveIterator(stream)):
        # 兼容各种 record.record_type 表示方式
        rec_type_obj = getattr(record, 'record_type', None)

        # 优先从 headers 读取 WARC-Type（通常是最可靠的字符串）
        warc_type_hdr = None
        try:
            # headers 可能是大小写不同的键
            if hasattr(record, 'headers'):
                warc_type_hdr = record.headers.get('WARC-Type') or record.headers.get('WARC-Type'.lower()) or record.headers.get('warc-type')
        except Exception:
            warc_type_hdr = None

        # 把 rec_type_obj 统一转换为字符串
        rec_type_str = ""
        if isinstance(rec_type_obj, str):
            rec_type_str = rec_type_obj
        else:
            # 尝试属性 name/value（枚举常用）
            rec_type_str = getattr(rec_type_obj, 'name', None) or getattr(rec_type_obj, 'value', None) or str(rec_type_obj)

        rec_type_str = (rec_type_str or "").lower()
        warc_type_hdr = (warc_type_hdr or "").lower()

        print(f"记录 #{i}  record.record_type(raw)={repr(rec_type_obj)}  -> rec_type_str='{rec_type_str}'  header WARC-Type='{warc_type_hdr}'")

        # 判断是否为 response：任一来源中包含 'response' 即可
        is_response = ('response' in rec_type_str) or ('response' in warc_type_hdr)

        # 读取内容（小心可能抛异常）
        try:
            content_bytes = record.reader.read()
        except Exception as e:
            print(f"  读取内容时出错: {e!r}")
            content_bytes = b""

        print("  content len:", len(content_bytes) if content_bytes is not None else "None")

        if is_response:
            found_response = True
            try:
                extracted = extract_text_from_html_bytes(content_bytes)
            except Exception as e:
                print("  extract_text_from_html_bytes 抛错：", repr(e))
                extracted = ""

            print("  URL header:", record.headers.get('WARC-Target-URI', None) if hasattr(record, 'headers') else None)
            print("  提取文本长度:", len(extracted) if extracted is not None else "None")
            print("-" * 40)
            print((extracted or "")[:1000])
            print("\n" + "=" * 80 + "\n")
            break

        if i >= max_records - 1:
            print(f"已遍历 {max_records} 条记录，未找到 response 类型记录。")
            break

if not found_response:
    print("在前", max_records, "条记录中未发现 type=response。")