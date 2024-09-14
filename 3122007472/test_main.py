import unittest
import numpy as np
from main import preprocess_text, segment_text, text_to_vector, calculate_cosine_similarity, calculate_edit_distance_similarity, similarity

class TestSimilarityModule(unittest.TestCase):

    # 测试1: 测试文本预处理函数 - 正常文本
    def test1_preprocess_text_normal(self):
        text = "今天是星期天，我去看电影了！"
        expected_result = "今天是星期天我去看电影了"
        self.assertEqual(preprocess_text(text), expected_result)

    # 测试2: 测试文本预处理函数 - 特殊字符
    def test2_preprocess_text_special_chars(self):
        text = "today是 .星期天！！"
        expected_result = "today是 星期天"
        print(preprocess_text(text))
        self.assertEqual(preprocess_text(text), expected_result)

    # 测试3: 测试文本预处理函数 - 空字符串
    def test3_preprocess_text_empty(self):
        text = ""
        expected_result = ""
        self.assertEqual(preprocess_text(text), expected_result)

    # 测试4: 测试jieba分词 - 正常文本
    def test4_segment_text_normal(self):
        text = "today is 星期天"
        result = segment_text(text)
        expected_words = ['today','is','星期', '天']  # 假设jieba分词结果
        print(result)
        self.assertGreater(len(result), 0)
        self.assertTrue(any(word in result for word in expected_words))

    # 测试5: 测试jieba分词 - 空字符串
    def test5_segment_text_empty(self):
        text = ""
        result = segment_text(text)
        self.assertEqual(result, [])

    # 测试6: 测试词频向量生成 - 正常文本
    def test6_text_to_vector_normal(self):
        words = ['今天', '是', '星期天']
        vocabulary = ['今天', '是', '星期', '天']
        vector = text_to_vector(words, vocabulary)
        expected_vector = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(vector, expected_vector)

    # 测试7: 测试词频向量生成 - 空文本
    def test7_text_to_vector_empty(self):
        words = []
        vocabulary = ['今天', '是', '星期', '天']
        vector = text_to_vector(words, vocabulary)
        expected_vector = np.array([0, 0, 0, 0])
        np.testing.assert_array_equal(vector, expected_vector)

    # 测试8: 测试余弦相似度 - 正常向量
    def test8_cosine_similarity_normal(self):
        vec1 = [1, 1, 0, 1]
        vec2 = [1, 1, 0, 0]
        similarity = calculate_cosine_similarity(vec1, vec2)
        expected_similarity = 0.816  # 预期结果
        self.assertAlmostEqual(similarity, expected_similarity, places=3)

    # 测试9: 测试余弦相似度 - 完全相同向量
    def test9_cosine_similarity_identical(self):
        vec1 = [1, 1, 1]
        vec2 = [1, 1, 1]
        similarity = calculate_cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0)

    # 测试10: 测试编辑距离相似度 - 相近文本
    def test10_edit_distance_similarity_close(self):
        text1 = "今天是星期天"
        text2 = "今天是周天"
        similarity = calculate_edit_distance_similarity(text1, text2)
        expected_similarity = 0.67  # 预期相似度
        self.assertAlmostEqual(similarity, expected_similarity, places=2)

    # 测试11: 测试编辑距离相似度 - 完全不同文本
    def test11_edit_distance_similarity_different(self):
        text1 = "今天是星期天"
        text2 = "明天是工作日"
        similarity = calculate_edit_distance_similarity(text1, text2)
        self.assertLess(similarity, 0.5)

    # 测试12: 测试编辑距离相似度 - 完全相同文本
    def test12_edit_distance_similarity_identical(self):
        text1 = "今天是星期天"
        text2 = "今天是星期天"
        similarity = calculate_edit_distance_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)

    # 测试13: 测试相似度函数 - 正常文本
    def test13_similarity_normal(self):
        text1 = "../examples/orig.txt"
        text2 = "../examples/orig_0.8_add.txt"
        similarity_result = similarity(text1, text2, cosine_weight=0.7, edit_distance_weight=0.3)
        self.assertGreater(similarity_result, 0.7)  # 预期相似度大于0.7

    # 测试14: 测试相似度函数 - 相同文本
    def test14_similarity_identical(self):
        text1 = "../examples/orig.txt"
        text2 = "../examples/orig.txt"
        similarity_result = similarity(text1, text2, cosine_weight=0.7, edit_distance_weight=0.3)
        self.assertAlmostEqual(similarity_result, 1.0)

    # 测试15: 测试相似度函数 - 不同文本
    def test15_similarity_different(self):
        text1 = "../examples/orig.txt"
        text2 = "../examples/orig_0.8_del.txt"
        similarity_result = similarity(text1, text2, cosine_weight=0.7, edit_distance_weight=0.3)
        self.assertAlmostEqual(similarity_result, 0.8)

if __name__ == '__main__':
    unittest.main()
