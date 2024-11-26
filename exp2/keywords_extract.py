import json


# in:list of captions(str)
# out: list of keywords(str)
def extract_keywords_tfidf(captions, keywords_num=5, debug=False):
    # 使用TF-IDF提取关键词
    # 假设英文，如果是中文需要移除stop_words参数
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english")

    keywords_str_list = []

    tfidf_matrix = vectorizer.fit_transform(captions)

    # 将TF-IDF矩阵转换为数组
    array = tfidf_matrix.toarray()

    # 为每句话提取关键词
    for idx, caption in enumerate(captions):
        tfidf_scores = zip(vectorizer.get_feature_names_out(), array[idx])
        sorted_tfidf = sorted(
            tfidf_scores, key=lambda x: x[1], reverse=True
        )  # 按TF-IDF分数降序排序
        # 选择TF-IDF分数最高的词作为关键词，这里选择3个作为示例
        keywords = [score[0] for score in sorted_tfidf[:keywords_num]]
        keywords_str = ", ".join(keywords)
        keywords_str_list.append(keywords_str)

    # 打印结果
    if debug:
        num = 3
        cnt = 0
        for captions, key_words in zip(caption, keywords):
            print(f"Sentence: {caption}")
            print(f"Keywords: {key_words}")
            print("\n")
            cnt += 1
            if cnt >= num:
                break
    return keywords_str_list


from typing import List, Tuple


def extract_keywords_keybert(captions: list, keywords_num=5, debug=False):
    """
    use keybert method
    """
    print("extract_keywords_keybert")
    import os

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from keybert import KeyBERT

    model = KeyBERT()
    keyword_pair_list_list: List[List[Tuple[str, float]]] = model.extract_keywords(
        captions,
        keyphrase_ngram_range=(15, 15),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=20,
        use_mmr=True,  # Maximal Margin Relevance
        diversity=0.7,
        top_n=keywords_num,
    )
    keywords_str_list = []
    for keyword_pair_list in keyword_pair_list_list:
        keywords_str = ", ".join([pair[0] for pair in keyword_pair_list[:keywords_num]])
        keywords_str_list.append(keywords_str)

    if debug:
        num = 3
        cnt = 0
        for caption, keywords_str in zip(captions, keywords_str_list):
            print(f"Sentence: {caption}")
            print(f"Keywords: {keywords_str}")
            print("--")
            cnt += 1
            if cnt >= num:
                break
    print("extract_keywords_keybert done")
    return keywords_str_list


from typing import Dict


def extract_keywords(
    infos: Dict[int, Dict],
    method: str = "gt",
    gt_json_path: str = "./dataset1/caption_with_keywords_and_image.json",
    keywords_num=5,
    debug=False,
):
    """
    infos: Dict[id -> info]
    info: Dict[str -> value], keys: "caption", "keywords", "image"
    """

    if method == "gt":
        # use ground truth keywords
        # same sequence ass infos.items()
        with open(gt_json_path, "r") as f:
            caption_with_keywords_and_image = json.load(f)
            keywords_str_list = [
                caption_with_keywords_and_image[i]["keywords"]
                for i in range(len(caption_with_keywords_and_image))
            ]
    elif method == "tfidf":
        # use tfidf method
        captions = [info["caption"] for info in infos.values()]
        keywords_str_list = extract_keywords_tfidf(
            captions, keywords_num=keywords_num, printKeywords=debug
        )
    elif method == "cutoff":
        # use cutoff method
        captions = [info["caption"] for info in infos.values()]
        keywords_str_list = [caption[:100] for caption in captions]
    elif method == "keybert":
        # use keybert method
        captions = [info["caption"] for info in infos.values()]
        keywords_str_list = extract_keywords_keybert(
            captions, keywords_num=keywords_num, debug=debug
        )
    else:
        print("method not supported")
        return

    for id, info in infos.items():
        info["keywords"] = keywords_str_list[id]
    return infos


if __name__ == "__main__":
    keywords = []
    sentences = []
    with open("./dataset1/caption.json", "r") as f:
        data = json.load(f)
        for i in range(len(data)):
            print(data[i]["caption"])
            print("\n")
            sentences.append(data[i]["caption"])
        print(sentences)
    extract_keywords_tfidf(sentences)
