
import os
import sys
import logging
from xml.etree.ElementTree import iterparse

import nltk
import nut2.gensim as gensim
import pdb

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

class XMLItemSentence(object):
    def __init__(self, source, tokenizer):
        """
        Simple format: one sentence = one text node in item node;
        words processed by NLTK. source a xml file object
        """
        self.source = source
        self.tokenizer = tokenizer

    def __iter__(self):
        """Iterate through the lines in the source."""

        # Assume it is a xml-like object and try treating it as such
        # Things that don't have seek will trigger an exception
        # self.source.seek(0)
        for event, elem in iterparse(self.source):
            if elem.tag == 'item':
                text = elem.find('text').text
                if text is None or len(text) < 1:
                    continue
                sentence = self.tokenizer.tokenize(text)
                # pdb.set_trace()
                if len(sentence) < 1:
                    continue
                yield sentence


def read_de_xml_file(item_xml_file):

    sentences = []
    wpt = nltk.tokenize.WordPunctTokenizer()
    count = 0
    for event, elem in iterparse(item_xml_file):
        if elem.tag == 'item':
            text = elem.find('text').text
            if text is None or len(text) < 1:
                continue
            sentence = wpt.tokenize(text)
            # pdb.set_trace()
            if len(sentence) < 1:
                continue
            sentences.append(sentence)
            count += 1
            if count % 1000 == 0:
                print count
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
    # sentences = read_de_xml_file(corpus_file_path)
    xis = XMLItemSentence(corpus_file_path, nltk.tokenize.WordPunctTokenizer())

    # print "%d sentence" % len(sentences)
    # train word2vec on the two sentences
    model = gensim.Word2Vec(xis, min_count=5)
    model.save(os.path.dirname(corpus_file_path) + '\mymodel', sep_limit=1000000)