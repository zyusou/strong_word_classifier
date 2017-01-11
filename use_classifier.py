import math
from gensim import corpora, models, matutils
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

num_dimension = 10


def make_word_feature(word):
    uni_gram = [char for char in word]
    bi_gram = [chars[0] + chars[1] for chars in zip(input_word, input_word[1:])]
    return uni_gram + bi_gram


def make_unit_vector(model, word_bow):
    lsi_vec = model[word_bow]
    dense_vec = list(matutils.corpus2dense([lsi_vec], num_terms=num_dimension).T[0])
    norm = math.sqrt(sum(num ** 2 for num in dense_vec))
    if norm == 0.0:
        unit_vec = [0.0 for num in dense_vec]
    else:
        unit_vec = [num / norm for num in dense_vec]

    return unit_vec


if __name__ == '__main__':
    if __name__ == '__main__':
        # 学習済みモデルや辞書をロード．
        strong_word_classifier = joblib.load("strong_word_classifier.jbl")
        """:type : LinearSVC"""

        dct = corpora.Dictionary.load_from_text(r"/Users/rin.kikuchi/PycharmProjects/"
                                                r"strong_word_classifier/train_data/"
                                                r"id2word.txt")
        """:type : corpora.dictionary.Dictionary'"""

        lsi_model = models.LsiModel.load("lsi_model.mod")

        accept_char = """
        アイウエオカキクケコサシスセソタチツテト
        ナニヌネノハヒフヘホマミムメモヤユヨ
        ラリルレロワヲンー
        ガギグゲゴザジズゼゾダヂヅデドバビブベボ
        パピプペポァィゥェォッュャョ
        """

        # 該当するカタカナ語を入力する．実際はHTTPから受け取る予定．今回は面倒なので決め打ち．
        # カタカナじゃなかったら，エラーを吐くようにする．
        #  Todo:実際にflask上で行うときにはコケたらマズいので，assert部分をUnity側で受け取れる何かを返す処理に書き直す必要あり
        input_word = "メタルギアソリッド"

        for input_char in input_word:
            assert input_char in accept_char, "カタカナ　イガイ　ヨワイ．"

        word_feature = make_word_feature(input_word)
        unit_vector = make_unit_vector(lsi_model, dct.doc2bow(word_feature))

        pred_label = strong_word_classifier.predict(unit_vector)
        score = strong_word_classifier.decision_function(unit_vector)

        print(pred_label, score)

        # 必殺技っぽい　→　マイナスになる．マイナス1近くなればよい
        # 実行時間，Mac上でだいたい0.005秒ぐらい．ミリ秒単位で動くなら十分では？
