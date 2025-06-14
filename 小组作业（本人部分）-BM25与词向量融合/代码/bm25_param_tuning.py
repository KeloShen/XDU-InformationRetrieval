from sklearn.datasets import fetch_20newsgroups
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import gensim.downloader as api

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 数据加载与预处理
def tokenize(text):
    return text.lower().split()

# 只选取内容相近的comp.*类别
categories = ['comp.graphics', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data
labels = newsgroups.target
corpus = [tokenize(doc) for doc in docs]

# 加载Google News预训练Word2Vec模型（首次运行需联网，较慢）
print('正在加载预训练Word2Vec模型...')
w2v_model = api.load('word2vec-google-news-300')
print('模型加载完成！')

# 计算文本的平均词向量
def get_avg_vector(tokens, model):
    vectors = [model[w] for w in tokens if w in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# BM25+Word2Vec加权检索
def bm25_w2v_search(query, bm25, w2v_model, docs_tokens, alpha=0.7, topk=50):
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    query_vec = get_avg_vector(tokenized_query, w2v_model)
    doc_vecs = np.array([get_avg_vector(doc, w2v_model) for doc in docs_tokens])
    w2v_sims = cosine_similarity([query_vec], doc_vecs)[0]
    final_scores = alpha * bm25_scores + (1 - alpha) * w2v_sims
    top_indices = np.argsort(final_scores)[::-1][:topk]
    return top_indices, final_scores

# 评估BM25和BM25+Word2Vec
np.random.seed(42)
query_num = 50
query_indices = np.random.choice(len(docs), query_num, replace=False)
queries = [docs[i] for i in query_indices]
true_labels = [labels[i] for i in query_indices]
bm25 = BM25Okapi(corpus)

topk = 50
# 原BM25
aps_bm25 = []
for i, query in enumerate(queries):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:topk]
    pred = [labels[idx] for idx in top_indices]
    y_true = [1 if l == true_labels[i] else 0 for l in pred]
    y_score = [scores[idx] for idx in top_indices]
    aps_bm25.append(average_precision_score(y_true, y_score))
map_bm25 = np.mean(aps_bm25)
print('原BM25 平均准确率（MAP）：', map_bm25)

# BM25+Word2Vec
aps_w2v = []
for i, query in enumerate(queries):
    top_indices, scores = bm25_w2v_search(query, bm25, w2v_model, corpus, alpha=0.7, topk=topk)
    pred = [labels[idx] for idx in top_indices]
    y_true = [1 if l == true_labels[i] else 0 for l in pred]
    y_score = [scores[idx] for idx in top_indices]
    aps_w2v.append(average_precision_score(y_true, y_score))
map_w2v = np.mean(aps_w2v)
print('BM25+Word2Vec 平均准确率（MAP）：', map_w2v)

# BM25+Word2Vec（alpha=0.3）
aps_w2v_03 = []
for i, query in enumerate(queries):
    top_indices, scores = bm25_w2v_search(query, bm25, w2v_model, corpus, alpha=0.3, topk=topk)
    pred = [labels[idx] for idx in top_indices]
    y_true = [1 if l == true_labels[i] else 0 for l in pred]
    y_score = [scores[idx] for idx in top_indices]
    aps_w2v_03.append(average_precision_score(y_true, y_score))
map_w2v_03 = np.mean(aps_w2v_03)
print('BM25+Word2Vec(alpha=0.3) 平均准确率（MAP）：', map_w2v_03)

# 多组对比图
plt.figure(figsize=(6,4))
plt.bar(['BM25', 'BM25+Word2Vec(0.7)', 'BM25+Word2Vec(0.3)'], [map_bm25, map_w2v, map_w2v_03], color=['skyblue', 'orange', 'green'])
plt.ylabel('MAP')
plt.title('不同方法检索效果对比')
for i, v in enumerate([map_bm25, map_w2v, map_w2v_03]):
    plt.text(i, v+0.002, f'{v:.4f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('bm25_vs_bm25w2v_multi.png')
plt.show() 