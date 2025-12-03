from collections import Counter
import os 
import hashlib

# uv run pytest -k test_exact_line_deduplication
def exact_line_deduplication(input_files, output_path):
    line_counts = Counter() # 相同的line只能出现一次
    os.makedirs(output_path, exist_ok=True)
    print("Pass 1: Counting lines...")
    for file in input_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line_hash = hashlib.md5(line.strip().encode('utf-8')).hexdigest()
                    line_counts[line_hash] += 1
        except Exception as e:
            print(f'Erro reading {file} {e}')
    print("Pass 2: Filtering and writing...")
    for file in input_files:
        file_name = os.path.basename(file)
        new_file = os.path.join(output_path, file_name)
        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as f_in, \
                open (new_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    line_content = line.strip()
                    if not line_content: 
                        # 空行跳过
                        continue 
                    line_hash = hashlib.md5(line_content.encode('utf-8')).hexdigest()
                    if line_counts[line_hash] > 1: continue
                    f_out.write(line)
        except Exception as e:
            print(f'Error writing {file} {e}')
    print('Finished !')

import re
import unicodedata
import numpy as np
from collections import defaultdict

# Step 1. Normalize
def normalize(text):
    text = unicodedata.normalize('NFD', text)
    text = text.lower()
    # 删掉 不是空白、不是字母、不是数字、不是下划线 的字符
    text = re.sub(r"[^\s\w]", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Step2. ngrams
def get_ngrams(text, n):
    words = text.split()
    if len(words) < n:
        return set()
    return set([' '.join(words[i: i + n]) for i in range(len(words) - n + 1)])

# Step 3 Jaccard_similarity
def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b: return 1.0
    if not set_a or not set_b: return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

# uv run pytest -k test_minhash_deduplication
def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    os.makedirs(output_directory, exist_ok=True)
    prime = (1 << 61) - 1
    np.random.seed(42) 
    # (a*x + b) % p 
    hash_params = np.random.randint(1, prime, size=(num_hashes, 2), dtype=np.uint64)
    docs = []

    print("Step 1: Computing Signatures...")
    for file in input_files:
        file_name = os.path.basename(file)
        with open(file, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        clean_text = normalize(text)
        doc_ngrams_set = get_ngrams(clean_text, ngrams)
        if not doc_ngrams_set: 
            signature = np.full(num_hashes, 2**64-1, dtype=np.uint64)
        else:
            ngrams_hash = np.array([hash(s) & 0xFFFFFFFF for s in doc_ngrams_set], dtype=np.uint64)
            raw_hash = (hash_params[:, 0:1] * ngrams_hash + hash_params[:, 1:2]) % prime
            signature = raw_hash.min(axis=1)
        docs.append({
            'path': file,
            'filename': file_name,
            'text': text,
            'ngrams': doc_ngrams_set,
            'signature': signature
        })
        
    print("Step 2: LSH Bucketing...")
    rows_per_band = num_hashes // num_bands
    candidate_pairs = set()
    for band_idx in range(num_bands):
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band
        buckets = defaultdict(list)
        for doc_idx, doc in enumerate(docs):
            sig = doc['signature']
            band_sig = tuple(sig[start_row: end_row])
            buckets[band_sig].append(doc_idx) # [(band1):[doc1, doc3, doc10, ...], (band2):[...]]
        # collision_count = sum(1 for docs in buckets.values() if len(docs) > 1)
        # print(f'Band {band_idx}: 发现 {collision_count} 组潜在重复。')
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i+1, len(bucket_docs)):
                        doc_a = bucket_docs[i]
                        doc_b = bucket_docs[j]
                        candidate_pairs.add(tuple(sorted((doc_a, doc_b)))) # doc_idx

    print(f"Step 3: Verifying {len(candidate_pairs)} candidates...")

    to_remove_indices = set()
    import networkx as nx
    G = nx.Graph()
    # 所有的文档作为节点，相似度高的两两连接
    for i in range(len(docs)):
        G.add_node(i)
    for (i, j) in candidate_pairs:
        siga, sigb = docs[i]['ngrams'], docs[j]['ngrams']
        if jaccard_similarity(siga, sigb) > jaccard_threshold:
            G.add_edge(i, j)

    print("Step 4: Clustering and Filtering...")
    components = list(nx.connected_components(G))
    # 所有联通分量list[set]
    # ex: [
    #       {'doc1', 'doc3', 'doc10'},
    #       {'doc2', 'doc8'}
    #      ]
    for comp in components:
        # 所有联通分量只留下一个（第一个）
        if len(comp) > 1:
            comp_list = sorted(list(comp))
            for idx in comp_list[1:]:
                to_remove_indices.add(idx)

    print("Step 5: Writing output...")
    for idx, doc in enumerate(docs):
        if idx not in to_remove_indices:
            output_path = os.path.join(output_directory, doc['filename'])
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc['text'])

    print(f"Removed {len(to_remove_indices)} duplicate documents.")
    print('Sucess!')


    







