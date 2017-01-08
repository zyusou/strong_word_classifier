import gensim

if __name__ == '__main__':

    with open(r"D:\Repository\untitled\train_data\false_data.txt",
              mode="r", encoding="utf-8") as false_file, \
            open(r"D:\Repository\untitled\train_data\true_data.txt",
                 mode="w", encoding="utf-8") as true_file:

        dict = gensim.corpora.Dictionary.load()