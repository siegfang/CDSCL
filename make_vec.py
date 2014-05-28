
import os
import sys
import logging
from xml.etree.ElementTree import iterparse

import nltk
import nut2.gensim as gensim

def read_en_xml_file(item_xml_file):

    sentences = []
    for event, elem in iterparse(item_xml_file):
        if elem.tag == 'item':
            text = elem.find('text').text
            sentence = text.strip().split(" ")
            if len(sentence) < 1:
                continue
            sentences.append(sentence)
    return sentences

def read_de_xml_file(item_xml_file):

    sentences = []
    for event, elem in iterparse(item_xml_file):
        if elem.tag == 'item':
            text = elem.find('text').text
            print text
            sentence = nltk.tokenize.WordPunctTokenizer().tokenize(text)
            if len(sentence) < 1:
                continue
            sentences.append(sentence)
    return sentences

def read_sentence(corpus_file_path):

    sentences = []
    with open(corpus_file_path, 'r') as corpus_file:
        for line in corpus_file:
            sentence = line.strip().split(" ")
            if len(sentence) < 1:
                continue
            sentences.append(sentence)
    return sentences

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus_file_path = sys.argv[1]

    # sentences = read_sentence(corpus_file_path)
    # sentences = read_en_xml_file(corpus_file_path)
    sentences = read_de_xml_file(corpus_file_path)


    print "%d sentence" % len(sentences)
    # train word2vec on the two sentences
    model = gensim.Word2Vec(sentences, min_count=1)
    model.save(os.path.dirname(corpus_file_path) + '\mymodel')