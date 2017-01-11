from gensim import corpora
from sklearn.externals import joblib

if __name__ == '__main__':
    # 学習済みモデルや辞書をロード．
    strong_word_classifier = joblib.load("strong_word_classifier.jbl")

    dct = corpora.Dictionary.load_from_text(r"/Users/rin.kikuchi/PycharmProjects/"
                                            r"strong_word_classifier/train_data/"
                                            r"id2word.txt")
    """:type : corpora.dictionary.Dictionary'"""

    accept_char = """
    アイウエオカキクケコサシスセソタチツテト
    ナニヌネノハヒフヘホマミムメモヤユヨ
    ラリルレロワヲンー
    ガギグゲゴザジズゼゾダヂヅデドバビブベボ
    パピプペポァィゥェォッュャョ
    """

    # 該当するカタカナ語を入力する．実際はHTTPから受け取る予定．今回は面倒なので決め打ち．
    # カタカナじゃなかったら，エラーを吐くようにする．
    #  Todo:実際にflask上で行うときにはコケたらマズいので，Unity側で受け取れる何かを返す用に処理を書き直す必要あり
    input_word = "ガラムマサラ"

    for char in input_word:
        assert char in accept_char, "カタカナ　イガイ　ヨワイ．"

    uni_gram = [char for char in input_word]
    bi_gram = [chars[0] + chars[1] for chars in zip(input_word, input_word[1:])]
    word_feature = uni_gram + bi_gram

    word_bow = dct.doc2bow(word_feature)

