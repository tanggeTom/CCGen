import os

def remove_index(path):
    with open(os.path.join(path, "test_1.gold_remove_index"), "a+") as fw:
        with open(os.path.join(path, "test_1.gold")) as f:
            for line in f.readlines():
                msg =line.split("\t")[1]
                fw.write(msg)
    with open(os.path.join(path, "test_1.output_remove_index"), "a+") as fw:
        with open(os.path.join(path, "test_1.output")) as f:
            for line in f.readlines():
                msg =line.split("\t")[1]
                fw.write(msg)

if __name__ == '__main__':
    remove_index("/home/tangzc/code2nl/model/dataseta")