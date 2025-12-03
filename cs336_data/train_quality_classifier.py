from cs336_data.preprocessing import extract_text_from_html_bytes, test_gopher
import os
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType


def prepare_training_data(pos_warc_path, neg_wet_path, output_path):
    print("正在生成训练数据...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 处理正样本
        count_pos = 0
        with open(pos_warc_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.record_type != WarcRecordType.response:
                    continue
                try:
                    content_bytes = record.reader.read()
                except Exception:
                    continue
                text = extract_text_from_html_bytes(content_bytes)
                if test_gopher(text) == False: continue
                clean_text = text.replace('\n', ' ')
                if len(clean_text.strip()) < 10:
                    continue
                f_out.write(f"__label__hq {clean_text}\n")
                count_pos += 1
        print(f'写入正样本: {count_pos}条')
        # 处理负样本
        count_neg = 0
        with open(neg_wet_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.record_type != WarcRecordType.conversion:
                    continue
                try:
                    text = record.reader.read().decode('utf-8', errors='replace')
                except Exception:
                    continue
                if test_gopher(text) == False: continue
                clean_text = text.replace('\n', ' ')
                if len(clean_text.strip()) < 10:
                    continue
                f_out.write(f"__label__cc {clean_text}\n")
                count_neg += 1
                if count_neg > count_pos: break 
        print(f'写入负样本: {count_neg}条')

def train_model(data_path, model_save_path):
    print("开始训练 FastText 模型...")
    # 训练监督模型
    model = fasttext.train_supervised(input=data_path, epoch=25, lr=1.0, wordNgrams=2)
    model.save_model(model_save_path)
    print(f"模型已保存至: {model_save_path}")


# 当前文件的运行绝对路径的目录
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    pos_warc_path = os.path.join(DIR_PATH, 'data', 'positive_samples.warc.gz')
    neg_wet_path = os.path.join(DIR_PATH, 'data', 'CC', 'CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz')
    output_path = os.path.join(DIR_PATH, 'data', 'quality_train.txt')
    model_path = os.path.join(DIR_PATH, 'data', 'classifiers', 'quality_classifier.bin')
    prepare_training_data(pos_warc_path, neg_wet_path, output_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        train_model(output_path, model_path)
    else:
        print("训练数据生成失败或为空，跳过训练。")

