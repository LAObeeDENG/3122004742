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
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 使用Jieba对文本进行分词
def segment_text(text):
    words = jieba.lcut(text)
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


def calculate_cosine_similarity(vec1, vec2):
    vec1 = [float(i) for i in vec1]
    vec2 = [float(i) for i in vec2]

    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(x ** 2 for x in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)




def main():
    original_file = "../examples/orig.txt"  # 获取命令行传入的原文文件路径
    modified_file = "../examples/orig_0.8_del.txt"  # 获取命令行传入的抄袭版文件路径
    output_file = "../examples/output.txt"  # 获取命令行传入的输出文件路径

    # 读取文件并预处理
    original_text = preprocess_text(read_file(original_file))
    modified_text = preprocess_text(read_file(modified_file))

    # Jieba 分词
    original_words = segment_text(original_text)
    modified_words = segment_text(modified_text)

    # 构建词汇表（词汇表基于两个文本的所有词）
    vocabulary = list(set(original_words + modified_words))

    # 将两个文本转化为词频向量
    original_vector = text_to_vector(original_words, vocabulary)
    modified_vector = text_to_vector(modified_words, vocabulary)

    # 计算余弦相似度
    similarity = calculate_cosine_similarity(original_vector, modified_vector)

    # 输出相似度结果
    # with open(output_file, 'w', encoding='utf-8') as file:
    print(f"文本查重：{similarity:.2f}")
        # file.write(f"{similarity:.2f}\n")

if __name__ == "__main__":
    main()
