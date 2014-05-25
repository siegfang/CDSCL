import util.bow as bow
import optparse
import argparse

__author__ = 'fangy'
__version__ = "2.0"


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """Prefixes `s_` and `t_` refer to source and target language
    , resp. Train and unlabeled files are expected to be in Bag-of-Words format.
    """
    parser = argparse.ArgumentParser(usage="%clscl_train [options] " \
                                           "s_lang t_lang s_train_file " \
                                           "s_unlabeled_file t_unlabeled_file " \
                                           "dict_file model_file",
                                     # version="%prog " + __version__,
                                     description=description)

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

    return parser


def train():
    """Training script for CLSCL.

    TODO: different translators.
    """
    parser = train_args_parser()
    args = parser.parse_args()
    if len(args) != 7:
        parser.error("incorrect number of arguments (use `--help` for help).")

    slang = args[0]
    tlang = args[1]

    fname_s_train = args[2]
    fname_s_unlabeled = args[3]
    fname_t_unlabeled = args[4]
    fname_dict = args[5]

    # Create vocabularies
    # s_voc = bow.vocabulary(fname_s_train, fname_s_unlabeled,
    #                        mindf=2,
    #                        maxlines=options.max_unlabeled)
    # t_voc = bow.vocabulary(fname_t_unlabeled,
    #                        mindf=2,
    #                        maxlines=options.max_unlabeled)
    # s_voc, t_voc, dim = bow.disjoint_voc(s_voc, t_voc)
    # s_ivoc = dict([(idx, term) for term, idx in s_voc.items()])
    # t_ivoc = dict([(idx, term) for term, idx in t_voc.items()])
    # print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    # print("|V| = %d" % dim)
    #
    # # Load labeled and unlabeled data
    # s_train, classes = bow.load(fname_s_train, s_voc, dim)
    # s_unlabeled, _ = bow.load(fname_s_unlabeled, s_voc, dim,
    #                           maxlines=options.max_unlabeled)
    # t_unlabeled, _ = bow.load(fname_t_unlabeled, t_voc, dim,
    #                           maxlines=options.max_unlabeled)
    # print("classes = {%s}" % ",".join(classes))
    # print("|s_train| = %d" % s_train.n)
    # print("|s_unlabeled| = %d" % s_unlabeled.n)
    # print("|t_unlabeled| = %d" % t_unlabeled.n)

    # Load dictionary
    # translator = DictTranslator.load(fname_dict, s_ivoc, t_voc)
    #
    # pivotselector = pivotselection.MISelector()
    # trainer = auxtrainer.ElasticNetTrainer(options.preg, options.alpha,
    #                                        10**6)
    # strategy_factory = {"hadoop": auxstrategy.HadoopTrainingStrategy,
    #                     "serial": auxstrategy.SerialTrainingStrategy,
    #                     "parallel": partial(auxstrategy.ParallelTrainingStrategy,
    #                                         n_jobs=options.n_jobs)}
    # clscl_trainer = CLSCLTrainer(s_train, s_unlabeled,
    #                              t_unlabeled, pivotselector,
    #                              translator, trainer,
    #                              strategy_factory[options.strategy](),
    #                              verbose=options.verbose)
    # clscl_trainer.s_ivoc = s_ivoc
    # clscl_trainer.t_ivoc = t_ivoc
    # model = clscl_trainer.train(options.m, options.phi, options.k)
    #
    # model.s_voc = s_voc
    # model.t_voc = t_voc
    # compressed_dump(argv[6], model)


if __name__ == "__main__":

    parser = train_args_parser()
    options, argv = parser.parse_args()
    print('\n'.join(argv))