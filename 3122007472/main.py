import sys
import re
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 文本预处理函数
def preprocess_text(text):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 去除多余的空白符
    text = re.sub(r'\s+', '', text).strip()
    return text

# 使用Jieba对文本进行分词
def segment_text(text):
    words = jieba.lcut(text,cut_all=True)
    return words

# 计算词频
def calculate_term_frequency(words):
    return Counter(words)


# 生成向量
def text_to_vector(words, vocabulary):
    vector = [0] * len(vocabulary)
    word_count = calculate_term_frequency(words)

    for word, count in word_count.items():
        if word in vocabulary:
            idx = vocabulary.index(word)
            vector[idx] = count
    return vector


#计算余弦相似度
def calculate_cosine_similarity(vec1, vec2):
    vec1 = [float(i) for i in vec1]
    vec2 = [float(i) for i in vec2]

    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(x ** 2 for x in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# 完整的余弦查重计算
def similarity(text1, text2):
    r_text1=read_file(text1)
    r_text2=read_file(text2)

    # 预处理
    p_text1 = preprocess_text(r_text1)
    p_text2 = preprocess_text(r_text2)

    # jieba分词
    original_words = segment_text(p_text1)
    modified_words = segment_text(p_text2)

    # 字典
    vocabulary = list(set(original_words + modified_words))
    print(vocabulary)
    # 将两个文本转化为词频向量
    original_vector = text_to_vector(original_words, vocabulary)
    modified_vector = text_to_vector(modified_words, vocabulary)
    print(original_vector)
    print(modified_vector)
    # 计算余弦相似度
    similarity_result = calculate_cosine_similarity(original_vector, modified_vector)
    return similarity_result


if __name__ == "__main__":
    text1 = "../examples/orig.txt"
    text2 = "../examples/orig_0.8_del.txt"
    similarity_result = similarity(text1, text2)
    print(f"文本查重：{similarity_result:.20f}")
