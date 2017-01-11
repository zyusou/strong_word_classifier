import joblib
from gensim import corpora, models, matutils
import math
from tqdm import tqdm
from pprint import pprint
import json

num_dimension = 10


def vec2dense(vector, num_terms):
    return list(matutils.corpus2dense([vector], num_terms=num_terms).T[0])


if __name__ == '__main__':
    processed_file_dict = joblib.load(r"/Users/rin.kikuchi/PycharmProjects/"
                                      r"strong_word_classifier/train_data/"
                                      r"processed_file.dict")
    """:type : dict"""

    dct = corpora.Dictionary.load_from_text(r"/Users/rin.kikuchi/PycharmProjects/"
                                            r"strong_word_classifier/train_data/"
                                            r"id2word.txt")
    """:type : corpora.dictionary.Dictionary'"""

    docs = {}
    bow_docs = {}

    # 最初の時点でもっとちゃんと辞書を作っておけばよかった気がする……．
    # 各単語のuni-gram, bi-gramをドキュメントに含まれる単語と見て学習させるので，
    # 各単語ごとに適当にキーを作って辞書にする．
    for key in processed_file_dict.keys():
        for i, doc in enumerate(processed_file_dict[key]):
            docs["".join([key, "_", str(i)])] = doc

        for doc_key in docs.keys():
            bow_docs[doc_key] = dct.doc2bow(docs[doc_key])

    lsi_docs = {}
    lsi_model = models.LsiModel(bow_docs.values(),
                                # id2word=dct.load_from_text("id2word.txt"),
                                id2word=dct,
                                num_topics=num_dimension)

    lsi_model.save("lsi_model.mod")

    # pprint(lsi_model.show_topics())
    # topic_weight_list = [[lsi_model.show_topic(topicno=topicno, topn=100)]
    #                      for topicno in range(num_dimension)]

    # C#で使えるようにするためにjsonでdump
    # for i, topic in enumerate(topic_weight_list):
    #     # pprint(topic[0])
    #     # print()
    #     with open(r"D:\Repository\untitled\json_data\{}".format("".join(["topic_", str(i), ".json"])),
    #               mode="w", encoding="utf-8") as json_file:
    #         weight_dict = {}
    #         for tup in topic[0]:  # (”トークン”, 値)の形で保存されているので，辞書にする
    #             weight_dict[str(tup[0])] = tup[1]
    #         json.dump(weight_dict, json_file, ensure_ascii=False)

    for key in bow_docs.keys():
        vec = bow_docs[key]
        lsi_vec = lsi_model[vec]
        lsi_docs[key] = lsi_vec

    unit_vectors = {}
    for key in tqdm(lsi_docs.keys(), total=len(lsi_docs.keys())):
        # num_termsの挙動が良く分からん……．LSI後の次元数(num_dimention)で正しいみたい．
        vec = vec2dense(lsi_docs[key], num_terms=num_dimension)
        norm = math.sqrt(sum(num ** 2 for num in vec))
        if norm == 0.0:
            unit_vector = [0.0 for num in vec]
        else:
            unit_vector = [num / norm for num in vec]

        unit_vectors[key] = unit_vector

    joblib.dump(unit_vectors, "train_data/unit_vectors.dict", compress=3)
