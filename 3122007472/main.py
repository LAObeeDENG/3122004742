import os
import sys
import re
import jieba
from collections import Counter
from Levenshtein import distance as levenshtein_distance  # 使用Levenshtein包计算编辑距离
from line_profiler_pycharm import profile

@profile
# 读取文件内容
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"没找到文件: {file_path}")

@profile
# 文本预处理函数
def preprocess_text(text):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 去除多余的空白符
    text = re.sub(r'\s+', ' ', text).strip()
    #去除停用词 长文本中的常用词不计入查重
    if len(text) > 800:
        text = re.sub(r'[的了是很我有和也吧啊你他她]','',text)
    return text

@profile
# 使用Jieba对文本进行分词
def segment_text(text):
    words = jieba.lcut(text, cut_all=True)
    processed_words = []
    for word in words:
        if word.strip():
            processed_words.append(word)
    return processed_words

@profile
# 计算词频
def calculate_term_frequency(words):
    return Counter(words)


@profile
# 生成向量
def text_to_vector(words, vocabulary):
    vector = [0] * len(vocabulary)
    word_count = calculate_term_frequency(words)

    for word, count in word_count.items():
        #词在字典中就把词频放入向量
        if word in vocabulary:
            idx = vocabulary.index(word)
            vector[idx] = count
    return vector

@profile
# 计算编辑距离相似度 更适合短文本 捕捉局部修改或顺序变化的影响
def calculate_edit_distance_similarity(text1, text2):
    edit_distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    return 1 - (edit_distance / max_len)

@profile
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

@profile
# 完整的余弦 + 编辑距离查重计算函数
def similarity(text1, text2,cosine_weight,edit_distance_weight):
    r_text1=read_file(text1)
    r_text2=read_file(text2)

    # 预处理
    p_text1 = preprocess_text(r_text1)
    p_text2 = preprocess_text(r_text2)

    if not p_text1 or not p_text2:
        raise ValueError("报错，空文件无法查重！请重新输入！")

    # jieba分词
    original_words = segment_text(p_text1)
    modified_words = segment_text(p_text2)

    # 字典
    vocabulary = list(set(original_words + modified_words))
    # 将两个文本转化为词频向量
    original_vector = text_to_vector(original_words, vocabulary)
    modified_vector = text_to_vector(modified_words, vocabulary)

    # 计算余弦相似度
    similarity_result = calculate_cosine_similarity(original_vector, modified_vector)
    print(f"余弦相似度：{similarity_result}")
    # 计算编辑距离相似度
    edit_distance_similarity_result = calculate_edit_distance_similarity(p_text1, p_text2)
    print(f"编辑距离相似度：{edit_distance_similarity_result}")
    # 加权计算最终相似度
    final_similarity = (cosine_weight * similarity_result) + (edit_distance_weight * edit_distance_similarity_result)
    return final_similarity

@profile
def main():
    if len(sys.argv) != 4:
        print("输入有误！用法: python main.py <original_file> <plagiarized_file> <output_file>")
        sys.exit(1)

    #控制台指令
    original_file = sys.argv[1]
    plagiarized_file = sys.argv[2]
    output_file = sys.argv[3]

    # 计算相似度
    similarity_result = similarity(original_file, plagiarized_file,
                                   cosine_weight=0.7, edit_distance_weight=0.3)
    print(f"文本查重：{similarity_result:.2f}")
    # 输出相似度结果到指定的文件中
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"文本查重：{similarity_result:.2f}\n")

if __name__ == "__main__":
    # text1 = "../examples/orig.txt"
    # text2 = "../examples/orig_0.8_dis_10.txt"
    # similarity_result = similarity(text1, text2,cosine_weight=0.7, edit_distance_weight=0.3)
    # print(f"文本查重：{similarity_result:.2f}")
    main()
