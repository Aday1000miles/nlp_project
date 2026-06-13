import re
import collections
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import jieba.posseg as pseg


# ==========================================
# 1. 载入停用词表
# ==========================================
def load_stopwords(filepath):
    print("正在加载停用词表...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f.readlines()])


# ==========================================
# 2. 文本清洗与分词核心逻辑
# ==========================================
def analyze_author_style(file_path, stopwords):
    print(f"正在深度分析文本: {file_path} ...")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 依据中文句末标点切分句子以统计句长
    raw_sentences = re.split(r'[。！？……\n]', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    total_words_in_text = 0
    sentence_count = len(sentences)

    filtered_words = []
    pos_tags = []

    for sentence in sentences:
        words_with_pos = pseg.lcut(sentence)
        total_words_in_text += len(words_with_pos)  # 原始分词总数用于句长

        for word, pos in words_with_pos:
            # 过滤停用词，且词长必须大于1（过滤掉单字和纯标点）
            if word not in stopwords and len(word) > 1:
                filtered_words.append(word)
                pos_tags.append(pos)

    avg_sentence_len = total_words_in_text / sentence_count if sentence_count > 0 else 0
    return filtered_words, pos_tags, avg_sentence_len


# ==========================================
# 3. 统计可视化与报告生成
# ==========================================
def generate_reports(words_A, pos_A, avg_len_A, words_B, pos_B, avg_len_B, name_A="鲁迅", name_B="刘鹗"):
    # 让 matplotlib 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 词频与词性频次统计
    counts_A = collections.Counter(words_A)
    counts_B = collections.Counter(words_B)

    pos_counts_A = collections.Counter(pos_A)
    pos_counts_B = collections.Counter(pos_B)

    ratio_A = {k: v / sum(pos_counts_A.values()) for k, v in pos_counts_A.items()}
    ratio_B = {k: v / sum(pos_counts_B.values()) for k, v in pos_counts_B.items()}

    # 画词云图
    print("生成词云图...")
    for words_dict, name, filename in [(counts_A, name_A, 'wc_A.png'), (counts_B, name_B, 'wc_B.png')]:
        wc = WordCloud(font_path='simhei.ttf', width=800, height=450, background_color='white')
        wc.generate_from_frequencies(words_dict)
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"{name} 作品高频词云", fontsize=18)
        plt.axis('off')
        plt.savefig(filename, dpi=300)
        plt.close()

    # 画词性对比柱状图
    print("生成词性对比图...")
    target_pos = ['n', 'v', 'a', 'd']  # 名、动、形、副
    pos_labels = {'n': '名词', 'v': '动词', 'a': '形容词', 'd': '副词'}

    plot_data = []
    for pos in target_pos:
        plot_data.append({'作者': name_A, '词性': pos_labels[pos], '占比': ratio_A.get(pos, 0)})
        plot_data.append({'作者': name_B, '词性': pos_labels[pos], '占比': ratio_B.get(pos, 0)})

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(9, 5))
    sns.barplot(x='词性', y='占比', hue='作者', data=df, palette='Set2')
    plt.title(f'{name_A} vs {name_B} 核心词性分布对比', fontsize=14)
    plt.savefig('pos_comparison.png', dpi=300)
    plt.close()

    # 控制台打印最终学术表格
    print("\n" + "=" * 40)
    print("          STYLOMETRY 最终文体计量结果")
    print("=" * 40)
    print(f"统计指标\t\t|\t{name_A}\t|\t{name_B}")
    print("-" * 40)
    print(f"过滤后有效词数\t|\t{len(words_A)}\t|\t{len(words_B)}")
    print(f"独立不同词数\t|\t{len(set(words_A))}\t|\t{len(set(words_B))}")
    print(f"词汇丰富度(TTR)\t|\t{len(set(words_A)) / len(words_A):.4f}\t|\t{len(set(words_B)) / len(words_B):.4f}")
    print(f"平均句长(词/句)\t|\t{avg_len_A:.2f}\t|\t{avg_len_B:.2f}")
    print("=" * 40)


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    stopwords = load_stopwords('stopwords.txt')

    # 开始数据运行
    words_A, pos_A, avg_len_A = analyze_author_style('A_SC.txt', stopwords)
    words_B, pos_B, avg_len_B = analyze_author_style('B_SC.txt', stopwords)

    # 生成数据图
    generate_reports(words_A, pos_A, avg_len_A, words_B, pos_B, avg_len_B, name_A="鲁迅", name_B="刘鹗")