from gensim import corpora
import joblib

# 高頻度，低頻度のトークンを削除するためのパラメータ．文字ベースだし控えめに．
no_below_value = 5        # n回未満出現したもの．デフォルトは5
no_above_value = 0.5      # 全体の文書中に出現する割合．文字ベースなので考慮しない．デフォルトは0.5


def make_char_list(file):
    documents = file.read().split("\n")

    char_list = [[char for char in line.split()]
                 for line in documents]

    return char_list


if __name__ == '__main__':
    with open(r"D:\Repository\untitled\train_data\false_data.txt",
              mode="r", encoding="utf-8") as false_file, \
            open(r"D:\Repository\untitled\train_data\true_data.txt",
                 mode="r", encoding="utf-8") as true_file:

        processed_file = {"false": make_char_list(false_file),
                          "true": make_char_list(true_file)}

    joblib.dump(processed_file, "train_data/processed_file.dict", compress=3)

    dct = corpora.Dictionary(processed_file["false"])
    dct.add_documents(processed_file["true"])

    dct.filter_extremes(no_below=no_below_value, no_above=no_above_value)

    dct.save_as_text("train_data/id2word.txt")
