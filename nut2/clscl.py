import util.bow as bow
import optparse
import argparse

__author__ = 'fangy'
__version__ = "2.0"


def train(arg_dict):
    """Training script for CLSCL.

    TODO: different translators.
    """

    slang = arg_dict['s_lang']
    tlang = arg_dict['t_lang']

    fname_s_train = arg_dict['s_labeled_file']
    fname_s_unlabeled = arg_dict['s_unlabeled_file']
    fname_t_unlabeled = arg_dict['t_unlabeled_file']
    fname_dict = arg_dict['dict_file']

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
