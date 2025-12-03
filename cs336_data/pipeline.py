import os 
import fasttext
from preprocessing import * 
import gzip
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from fastwarc.warc import ArchiveIterator, WarcRecordType

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型目录
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'classifiers')
# 模型地址
LID_MODEL_PATH = os.path.join(MODEL_DIR, 'lid.176.bin')
NSFW_MODEL_PATH = os.path.join(MODEL_DIR, 'jigsaw_fasttext_bigrams_nsfw_final.bin')
TOXIC_MODEL_PATH = os.path.join(MODEL_DIR, 'jigsaw_fasttext_bigrams_hatespeech_final.bin')
QUALITY_MODEL_PATH = os.path.join(MODEL_DIR, 'quality_classifier.bin')

models = {}
def init_worker():
    """在每个进程启动时加载模型，避免重复加载或 Pickling 问题"""
    global models
    fasttext.FastText.eprint = lambda x: None
    if os.path.exists(LID_MODEL_PATH):
        models['lid'] = fasttext.load_model(LID_MODEL_PATH)
    if os.path.exists(NSFW_MODEL_PATH):
        models['nsfw'] = fasttext.load_model(NSFW_MODEL_PATH)
    if os.path.exists(TOXIC_MODEL_PATH):
        models['toxic'] = fasttext.load_model(TOXIC_MODEL_PATH)
    if os.path.exists(QUALITY_MODEL_PATH):
        models['quality'] = fasttext.load_model(QUALITY_MODEL_PATH)

def predict_fasttext(model_key, text):
    if model_key not in models: return None, 0.0
    text = text.replace('\n', ' ')
    labels, scores = models[model_key].predict(text)
    label = labels[0].replace('__label__', '')
    score = scores[0]
    return label, score

# 处理单个文件
def process_wet_file(args):
    input_path, output_path = args
    logging = Counter()
    try:
        with open(input_path, 'rb') as stream, \
        gzip.open(output_path, 'wt') as f_out:
            for record in ArchiveIterator(stream):
                if record.record_type != WarcRecordType.conversion:
                    continue
                logging['total_docs'] += 1
                try:
                    # WET 已经是提取好的文本，但在 fastwarc 中需要 decode回text
                    text = record.reader.read().decode('utf-8', errors='replace')
                except:
                    logging['read_error'] += 1
                    continue
                if not text.strip():
                    logging['empty'] += 1
                    continue
                # 1. Gopher
                if not test_gopher(text):
                    logging['rejected_gopher'] += 1
                    continue
                # 2. Launguage
                lang, score = predict_fasttext('lid', text)
                if lang != 'en' or score < 0.6:
                    logging['rejected_lang'] += 1
                    continue
                # 3. Harmful
                nsfw_label, nsfw_score = predict_fasttext('nsfw', text)
                if nsfw_label == 'nsfw' and nsfw_score > 0.6: 
                    logging['rejected_nsfw'] += 1
                    continue
                # Toxic
                toxic_label, toxic_score = predict_fasttext('toxic', text)
                if toxic_label == 'toxic' and toxic_score > 0.6: 
                    logging['rejected_toxic'] += 1
                    continue
                # 4. Quality
                qual_label, qual_score = predict_fasttext('quality', text)
                # 如果确信是 'cc' (垃圾)就丢掉。
                if qual_label == 'cc' and qual_score > 0.5: 
                    logging['rejected_quality_model'] += 1
                    continue
                text, _ = mask_emails(text)
                text, _ = mask_ips(text)
                text, _ = mask_phone_numbers(text)
                f_out.write(text.strip() + "\n\n")
                logging['kept'] += 1
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return logging
    
    return logging
        


# CPU并行处理一堆文件
def main():
    INPUT_DIR = os.path.join(BASE_DIR, 'data', 'CC')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'filtered-0.5')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 扫描得到所有 WET 文件
    wet_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wet.gz')]
    
    tasks = []
    for f in wet_files:
        in_path = os.path.join(INPUT_DIR, f)
        out_path = os.path.join(OUTPUT_DIR, f.replace('.wet.gz', '.filtered.txt.gz'))
        tasks.append((in_path, out_path))

    print(f"找到 {len(tasks)} 个 WET 文件，准备处理...")
    
    # 并行处理
    total_stats = Counter()
    # initializer只用加载一次模型，避免Pickling错误
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
        futures = {executor.submit(process_wet_file, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks)):
            file_stats = future.result()
            total_stats += file_stats

    print("\n" + "="*30)
    print("过滤统计报告")
    print("="*30)
    for k, v in total_stats.items():
        print(f"{k}: {v}")
    
    if total_stats['total_docs'] > 0:
        kept_ratio = total_stats['kept'] / total_stats['total_docs'] * 100
        print(f"保留率: {kept_ratio:.2f}%")

if __name__ == "__main__":
    main()


'''Testing threshold'''       
# qual_score > 0.9
#  ==============================
# 过滤统计报告
# ==============================
# total_docs: 27173
# rejected_gopher: 10766
# rejected_quality_model: 6176
# rejected_lang: 9412
# kept: 781
# rejected_nsfw: 20
# rejected_toxic: 18
# 保留率: 2.87%


# qual_score > 0.5
# ==============================
# 过滤统计报告
# ==============================
# total_docs: 27173
# rejected_gopher: 10766
# rejected_quality_model: 6681
# rejected_lang: 9412
# rejected_nsfw: 20
# rejected_toxic: 18
# kept: 276
# 保留率: 1.02%

