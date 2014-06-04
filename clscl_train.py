#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

import nut2.clscl
# from multiprocessing import freeze_support # use when run in Windows

import argparse

def get_train_arg():

     description = """Prefixes `s_` and `t_` refer to source and target language
         , resp. Train and unlabeled files are expected to be in Bag-of-Words format.
         """
     parser = argparse.ArgumentParser(# version="%prog " + __version__,
                                      description=description)

     parser.add_argument('s_lang',
                         action="store",
                         help="source language")

     parser.add_argument('t_lang',
                         action="store",
                         help="target language")

     parser.add_argument('s_labeled_file',
                         action="store",
                         help="source language labelled file path")

     parser.add_argument('s_unlabeled_file',
                         action="store",
                         help="source language unlabelled file path")

     parser.add_argument('t_unlabeled_file',
                         action="store",
                         help="target language unlabelled file path")

     parser.add_argument('s_word2vec_file',
                         action="store",
                         help="the path stored the source word2vec model")

     parser.add_argument('t_word2vec_file',
                         action="store",
                         help="the path stored the target word2vec model")

     parser.add_argument('dict_file',
                         action="store",
                         help="source to target language dictionary file path")

     parser.add_argument('model_file',
                         action="store",
                         help="the path stored the model")

     parser.add_argument("-v", "--verbose",
                         dest="verbose",
                         help="verbose output",
                         default=1,
                         metavar="[0,1,2]",
                         type=int)

     parser.add_argument("-k",
                         dest="k",
                         help="dimensionality of cross-lingual representation.",
                         default=100,
                         metavar="int",
                         type=int)

     parser.add_argument("-m",
                         dest="m",
                         help="number of pivots.",
                         default=450,
                         metavar="int",
                         type=int)

     parser.add_argument("--max-unlabeled",
                         dest="max_unlabeled",
                         help="max number of unlabeled documents to read;" \
                              "-1 for unlimited.",
                         default=-1,
                         metavar="int",
                         type=int)

     parser.add_argument("-p", "--phi",
                         dest="phi",
                         help="minimum support of pivots.",
                         default=30,
                         metavar="int",
                         type=int)

     parser.add_argument("-r", "--pivot-reg",
                         dest="preg",
                         help="regularization parameter lambda for " \
                              "the pivot classifiers.",
                         default=0.00001,
                         metavar="float",
                         type=float)

     parser.add_argument("-a", "--alpha",
                         dest="alpha",
                         help="elastic net hyperparameter alpha.",
                         default=0.85,
                         metavar="float",
                         type=float)

     parser.add_argument("--strategy",
                         dest="strategy",
                         help="The strategy to compute the pivot classifiers." \
                              "Either 'serial' or 'parallel' [default] or 'hadoop'.",
                         default="parallel",
                         metavar="str",
                         type=str)

     parser.add_argument("--n-jobs",
                         dest="n_jobs",
                         help="The number of processes to fork." \
                              "Only if strategy is 'parallel'.",
                         default=-1,
                         metavar="int",
                         type=int)
     arg_dict = vars(parser.parse_args())
     return arg_dict

if __name__ == '__main__':
     arg_dict = get_train_arg()

     # freeze_support()  # use when run in Windows
     nut2.clscl.train(arg_dict)
