import gzip
import random
from fastwarc.warc import ArchiveIterator
from preprocessing import extract_text_from_html_bytes, identify_language

warc_path = "cs336_data/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def analyze_warc_languages():
    records_processed = 0
    english_count = 0
    samples_to_inspect = []
    
    # 随机采样的概率，避免只看到文件开头的记录
    sample_rate = 0.1 

    print(f"Reading {warc_path}...")
    
    try:
        with gzip.open(warc_path, 'rb') as stream:
            # 流式读取
            for record in ArchiveIterator(stream):
                # 考虑response内容
                if record.record_type.name == 'response':
                    # 随机抽取
                    if random.random() > sample_rate: continue
                    content_bytes = record.reader.read()
                    text = extract_text_from_html_bytes(content_bytes)
                    if not text.strip():
                        continue
                    lang, score = identify_language(text)
                    
                    records_processed += 1
                    if lang == 'en':
                        english_count += 1
                    
                    # 3. 收集 20 个样本进行人工检查
                    if len(samples_to_inspect) < 20:
                         samples_to_inspect.append((text, lang, score))
                    else:
                        # 简单的随机替换，保证看后面的一些数据
                        if random.random() < 0.05:
                            idx = random.randint(0, 19)
                            samples_to_inspect[idx] = (text, lang, score)

                    if records_processed >= 1000: 
                        break

        # 打印统计结果
        print(f"\n--- Statistics (based on first {records_processed} docs) ---")
        if records_processed > 0:
            print(f"English documents: {english_count} ({english_count/records_processed:.2%})")
        else:
            print("Warning: No response records found.")
        
        
        print("\n--- Manual Inspection of 20 Examples ---")
        for i, (text, lang, score) in enumerate(samples_to_inspect):
            print(f"Example {i+1}:")
            print(f"Predicted: [{lang}] (Score: {score:.4f})")
            # 打印文本的前200个字符，把换行替换掉方便阅读
            snippet = text[:200].replace('\n', ' ')
            print(f"Text: {snippet}...") 
            print("-" * 50)

    except FileNotFoundError:
        print("Error: WARC file not found. Please check the path.")

if __name__ == "__main__":
    analyze_warc_languages()