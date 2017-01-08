import os

if __name__ == '__main__':
    before_corpus_dir = "pure_corpus"
    after_corpus_dir = "preprocessed_corpus"
    acccept_char = """
    アイウエオカキクケコサシスセソタチツテト
    ナニヌネノハヒフヘホマミムメモヤユヨ
    ラリルレロワヲンー
    ガギグゲゴザジズゼゾダヂヅデドバビブベボ
    パピプペポァィゥェォッュャョ
    """

    for file_name in os.listdir(before_corpus_dir):
        before_path = os.path.join(before_corpus_dir, file_name)
        after_path = os.path.join(after_corpus_dir, file_name)

        with open(before_path, mode="r", encoding="utf-8") as before_file, \
                open(after_path, mode="w", encoding="utf-8") as after_file:

            for line in before_file:
                write_word = ""
                for char in line.strip():
                    if char in acccept_char:
                        write_word += char

                if len(write_word) != 0:
                    after_file.write(write_word + "\n")
