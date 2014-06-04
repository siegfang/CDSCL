#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

import nut2.clscl
import argparse

def get_predict_arg():
    """Create argument and option parser for the
    prediction script.
    """
    description = """Prefixes `s_` and `t_` refer to source and target language
    , resp. Train and unlabeled files are expected to be in Bag-of-Words format.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('s_labeled_file',
                        action="store",
                        help="source language labelled file path")

    parser.add_argument('model_file',
                        action="store",
                        help="the path stored the model")

    parser.add_argument('t_test_file',
                        action="store",
                        help="target language test file path")

    parser.add_argument('s_word2vec_file',
                         action="store",
                         help="the path stored the source word2vec model")

    parser.add_argument('t_word2vec_file',
                         action="store",
                         help="the path stored the target word2vec model")


    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        help="verbose output",
                        default=1,
                        metavar="[0,1,2]",
                        type=int)

    parser.add_argument("-R", "--repetition",
                        dest="repetition",
                        help="Repeat training `repetition` times and " \
                        "report avg. error.",
                        default=10,
                        metavar="int",
                        type=int)

    parser.add_argument("-r", "--reg",
                        dest="reg",
                        help="regularization parameter lambda. ",
                        default=0.01,
                        metavar="float",
                        type=float)

    parser.add_argument("--n-jobs",
                        dest="n_jobs",
                        help="The number of processes to fork.",
                        default=1,
                        metavar="int",
                        type=int)

    arg_dict = vars(parser.parse_args())
    return arg_dict

if __name__ == '__main__':

    nut2.clscl.predict(get_predict_arg())
