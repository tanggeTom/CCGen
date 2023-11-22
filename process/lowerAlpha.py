import os

from nltk.stem import WordNetLemmatizer
import nltk


def normalize_word(sentence):
    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    res = []
    for word, pos in pos_tags:
        if pos.startswith('J'):
            tmp = lemmatizer.lemmatize(word.lower(), 'a')
        elif pos.startswith('V'):
            tmp = lemmatizer.lemmatize(word.lower(), 'v')
        elif pos.startswith('N'):
            tmp = lemmatizer.lemmatize(word.lower(), 'n')
        elif pos.startswith('R'):
            tmp = lemmatizer.lemmatize(word.lower(), 'r')
        else:
            tmp = lemmatizer.lemmatize(word.lower())
        res.append(tmp)
    return ' '.join(res)
path = "model/1data_900_r_200_30"
if __name__ == '__main__':
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    with open(os.path.join(path, "test_1.gold_remove_index"),"r") as f, open(os.path.join(path, "test_1.goldlow"),"w") as f2:
        for i in f:
            i = i.strip()
            if i =='':
                continue
            f2.write(normalize_word(i))
            f2.write("\n")
    with open(os.path.join(path, "test_1.output_remove_index"), "r") as f, open(os.path.join(path, "test_1.outputlow"), "w") as f2:
        for i in f:
            i = i.strip()
            if i == '':
                continue
            f2.write(normalize_word(i))
            f2.write("\n")
