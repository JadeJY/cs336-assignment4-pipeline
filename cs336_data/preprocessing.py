import resiliparse.extract.html2text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import os

# uv run pytest -k test_extract_text_from_html_bytes 
def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    if not html_bytes: return ""
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        decoding_type = detect_encoding(html_bytes)
        html_str = html_bytes.decode(decoding_type, errors='replace')
    text = resiliparse.extract.html2text.extract_plain_text(
        html_str,        
    )
    return text

language_model_path = "cs336_data/data/classifiers/lid.176.bin"
language_model = fasttext.load_model(language_model_path)
# uv run pytest -k test_identify_language
def identify_language(text: str) -> tuple[str, float]:
    if not text or not text.strip(): return None
    clean_text = text.replace('\n', ' ')
    label_list, score_list = language_model.predict(clean_text, k=1)
    label, score = label_list[0], score_list[0]
    # label: __label__en
    lang_label = label.replace("__label__", "")
    return lang_label, score

# uv run pytest -k test_mask_emails
import re 
def mask_emails(string: str) -> tuple[str, int]:
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    new_text, num = re.subn(email_pattern, "|||EMAIL_ADDRESS|||", string)
    return new_text, num 

# uv run pytest -k test_mask_phones
def mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_pattern = r'(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}'
    new_text, count = re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)
    return new_text, count


# uv run pytest -k test_mask_ips
def mask_ips(text: str) -> tuple[str, int]:
    ip_pattern = r'\b(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}\b'
    new_text, count = re.subn(ip_pattern, "|||IP_ADDRESS|||", text)
    return new_text, count

# 加载模型
base_dir = os.path.dirname(os.path.abspath(__file__)) # 当前文件的运行绝对路径的目录
nsfw_model_path = os.path.join(base_dir, "data", "classifiers", "jigsaw_fasttext_bigrams_nsfw_final.bin")
nsfw_model = fasttext.load_model(nsfw_model_path)
hate_model_path = os.path.join(base_dir, "data", "classifiers", "jigsaw_fasttext_bigrams_hatespeech_final.bin")
hate_model = fasttext.load_model(hate_model_path)

# uv run pytest -k test_classify_nsfw
def classify_nsfw_speech(text: str) -> tuple[str, float]:
    clean_text = text.replace('\n', ' ')
    labels, scores = nsfw_model.predict(clean_text, k=1)
    label = labels[0].replace('__label__', '')
    score = scores[0]
    return label, score 


# uv run pytest -k test_classify_toxic_speech
def classify_toxic_speech(text: str) -> tuple[str, float]:
    clean_text = text.replace('\n', ' ')
    labels, scores = hate_model.predict(clean_text, k=1)
    label = labels[0].replace('__label__', '')
    score = scores[0]
    return label, score 

# uv run pytest -k test_gopher
import nltk
def test_gopher(text: str) -> bool:
    try:
        words = nltk.word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        words = nltk.word_tokenize(text)
    # Rule 1
    num_words = len(words)
    if num_words < 50 or num_words > 100000:
        return False 
    # Rule 2
    total_characters = sum(len(w) for w in words)
    avg_characters = total_characters / num_words
    if not (3 <= avg_characters <= 10):
        return False
    # Rule 3
    lines = text.splitlines()
    if len(lines) > 0:
        target_lines_num = sum(1 for line in lines if line.strip().endswith('...'))
        if target_lines_num / len(lines) > 0.3: return False
    alphabets_num = sum(1 for w in words if any(c.isalpha() for c in w))
    if alphabets_num / num_words < 0.8: return False
    return True 

# uv run pytest -k test_classify_quality
    



