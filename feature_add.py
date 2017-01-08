
import os

if __name__ == '__main__':

    before_corpus_dir = "preprocessed_corpus"
    after_corpus_dir = "feature_add_corpus"

    for file_name in os.listdir(before_corpus_dir):
        before_path = os.path.join(before_corpus_dir, file_name)
        after_path = os.path.join(after_corpus_dir, file_name)

        with open(before_path, mode="r", encoding="utf-8") as before_file, \
                open(after_path, mode="w", encoding="utf-8") as after_file:

            for line in before_file:
                word = line.strip()
                uni_gram = " ".join(word)
                bi_gram = " ".join([chars[0] + chars[1] for chars in zip(word, word[1:])])
                after_file.write(" ".join([uni_gram, bi_gram]) + "\n")

