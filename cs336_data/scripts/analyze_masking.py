import gzip
import re, random
from fastwarc.warc import ArchiveIterator
from preprocessing import extract_text_from_html_bytes, mask_ips, mask_emails, mask_phone_numbers

warc_path = "cs336_data/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
def run_inspection(warc_path, target_examples=20):
    print(f"正在读取文件: {warc_path}...")
    found_count = 0 
    sample_rate = 0.1 
    try:
        with gzip.open(warc_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if found_count >= target_examples:
                    break
                if record.record_type.name != 'response':
                    continue
                if random.random() > sample_rate: continue
                try:
                    content_bytes = record.reader.read()
                except Exception:
                    continue
                text = extract_text_from_html_bytes(content_bytes)
                if not text or not text.strip():
                    continue
                masked_text_email, e_count = mask_emails(text)
                masked_text_phone, p_count = mask_phone_numbers(text)
                masked_text_ip, i_count = mask_ips(text)
                total_hits = e_count + p_count + i_count

                if total_hits > 0:
                    found_count += 1
                    print(f"\n[{found_count}/{target_examples}] URI: {record.headers.get('WARC-Target-URI')}")
                    print("-" * 60)

                    # 打印 Email 命中情况
                    if e_count > 0:
                        print(f"Found {e_count} EMAILS:")
                        # 简单的 diff 显示逻辑
                        show_context(text, "|||EMAIL_ADDRESS|||", masked_text_email)
                    
                    # 打印 Phone 命中情况
                    if p_count > 0:
                        print(f"Found {p_count} PHONES:")
                        show_context(text, "|||PHONE_NUMBER|||", masked_text_phone)

                    # 打印 IP 命中情况
                    if i_count > 0:
                        print(f"Found {i_count} IPS:")
                        show_context(text, "|||IP_ADDRESS|||", masked_text_ip)
                    
                    print("=" * 60)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {warc_path}，请确认路径。")

def show_context(original_text, tag, masked_text):
    # 这里我们简化处理：直接打印出包含被替换特征的原始文本片段
    # 为了准确找到原词，我们需要反向推导或者正则搜索（为了演示简单，我们再次用正则搜出原词）
    if tag == "|||EMAIL_ADDRESS|||":
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    elif tag == "|||PHONE_NUMBER|||":
        pattern = r'(?:\+?1[-. ]?)?\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
    elif tag == "|||IP_ADDRESS|||":
        pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    else:
        print(f"Unknown tag: {tag}")
        return
    # 2. 在原文中查找所有匹配项
    matches = list(re.finditer(pattern, original_text))
    for i, match in enumerate(matches):
        start = max(0, match.start() - 30)
        end = min(len(original_text), match.end() + 30)
        snippet = original_text[start:end].replace('\n', ' ')
        matched_str = match.group()
        tag_index = masked_text.find(tag)
        if tag_index != -1:
            m_start = max(0, tag_index - 30)
            m_end = min(len(masked_text), tag_index + len(tag) + 30)
            masked_snippet = masked_text[m_start:m_end].replace('\n', ' ')
        else:
            masked_snippet = "(Could not locate tag in masked text)"
        print(f"   [Example #{i+1}]")
        print(f"   Context: ...{snippet}...")
        print(f"   Matched: '{matched_str}' -> Replaced with {tag}")
        print(f"   Masked Context  : ...{masked_snippet}...")
        print("-" * 30)
        
if __name__ == "__main__":
    run_inspection(warc_path)