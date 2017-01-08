import joblib
from gensim import corpora, models, matutils
import math

num_dimension = 10


def vec2dense(vector, num_terms):
    return list(matutils.corpus2dense([vector], num_terms=num_terms).T[0])


if __name__ == '__main__':
    processed_file_dict = joblib.load(r"D:\Repository\untitled\train_data\processed_file.dict")
    """:type : dict"""

    dct = corpora.Dictionary.load_from_text(r"D:\Repository\untitled\train_data\id2word.txt")
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

    for key in bow_docs.keys():
        vec = bow_docs[key]
        lsi_vec = lsi_model[vec]
        lsi_docs[key] = lsi_vec

    unit_vectors = {}
    for key in lsi_docs.keys():
        vec = vec2dense(lsi_docs[key], num_terms=num_dimension)
        norm = math.sqrt(sum(num ** 2 for num in vec))
        unit_vector = [num / norm for num in vec]
        unit_vectors[key] = unit_vector

    joblib.dump(unit_vectors, "train_data/unit_vectors.dict", compress=3)
