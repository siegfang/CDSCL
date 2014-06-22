# coding=utf-8
__author__ = 'fangy'

from xml.etree.ElementTree import iterparse
from collections import defaultdict


def read_corpus(corpus_file_path, sections=['text']):
    for event, elem in iterparse(corpus_file_path):
        if elem.tag == 'item':
            values = [elem.find(section).text for section in sections]
            if not all(values):
                continue

            rating_text = elem.find('rating')
            if rating_text is not None:
                rating_text = rating_text.text
                rating = float(rating_text.strip())
                if rating < 3:
                    label = 0
                else:
                    label = 1
            else:
                rating_text = elem.find('polarity')
                if rating_text is None:
                    label = -1
                elif rating_text.text.strip() == 'N':
                    label = 0
                else:
                    label = 1

            yield values, label


def read_corpora(corpus_file_paths, sections=['text']):
    for corpus_file_path in corpus_file_paths:
        # pdb.set_trace()
        for values, label in read_corpus(corpus_file_path, sections):
            yield values, label
            # print corpus_file_path


def transfer(source_file_path, target_file_path):

    target_file = open(target_file_path, 'w')

    for values, label in read_corpus(source_file_path):
        text = values[0]
        text = text.replace("\n", "")
        text = text.replace("  ", "")  # 替换回车产生的双空格
        sentence = text.split(' ')
        d = defaultdict(lambda: 0)
        for w in sentence:
            d[w] += 1
        sample = ' '.join([w + ":" + str(count) for w, count in d.items()]).encode('utf-8') + " "
        target_file.write(sample)

        if label > 0.5:
            target_file.write("#label#:positive\n")
        elif label > -0.5:
            target_file.write("#label#:negative\n")
        else:
            target_file.write("#label#:unknown\n")

    target_file.close()

def transfer_train(source_file_path, target_file_path, count=2000):

    if count > 2000:
        count = 2000
    pos_count = count/2
    neg_count = count - pos_count

    target_file = open(target_file_path, 'w')

    for values, label in read_corpus(source_file_path):
        text = values[0]
        text = text.replace("\n", "")
        text = text.replace("  ", "")  # 替换回车产生的双空格
        sentence = text.split(' ')
        d = defaultdict(lambda: 0)
        for w in sentence:
            d[w] += 1
        sample = ' '.join([w + ":" + str(count) for w, count in d.items()]).encode('utf-8') + " "

        if label > 0.5 and pos_count > 0:
            target_file.write(sample+"#label#:positive\n")
            pos_count -= 1
        elif label > -0.5 and neg_count > 0:
            target_file.write(sample+"#label#:negative\n")
            neg_count -= 1
        elif pos_count < 0 and neg_count < 0:
            break

    target_file.close()



if __name__ == '__main__':
    file_type = "train"
    source_file_path = "/Users/fangy/data/cl14-unprocessed/cn/music/" + file_type + ".review"
    target_file_path = "/Users/fangy/data/cl14-processed/cn/music/" + file_type + ".processed"
    # transfer(source_file_path, target_file_path)

    transfer_train(source_file_path, target_file_path, count=1500)
