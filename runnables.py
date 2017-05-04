#!/usr/bin/env python3
"""
A module to run different recommenders.
"""
import sys
import itertools
import numpy
import time
from optparse import OptionParser
from lib.evaluator import Evaluator
from lib.collaborative_filtering import CollaborativeFiltering
from lib.grid_search import GridSearch
from lib.LDA import LDARecommender
from lib.LDA2Vec import LDA2VecRecommender
from lib.SDAE import SDAERecommender
from lib.recommender_system import RecommenderSystem
from util.abstracts_preprocessor import AbstractsPreprocessor
from util.data_parser import DataParser
from util.recommender_configuer import RecommenderConfiguration
from util.model_initializer import ModelInitializer
from util.runs_loader import RunsLoader


class RunnableRecommenders(object):
    """
    A class that is used to run recommenders.
    """
    def __init__(self, use_database=True, verbose=True, load_matrices=True, dump=True, train_more=True,
                 random_seed=False, config=None):
        """
        Setup the data and configuration for the recommenders.
        """
        if use_database:
            self.ratings = numpy.array(DataParser.get_ratings_matrix())
            self.documents, self.users = self.ratings.shape
            self.abstracts_preprocessor = AbstractsPreprocessor(DataParser.get_abstracts(),
                                                                *DataParser.get_word_distribution())
        else:
            abstracts = {0: 'hell world berlin dna evolution', 1: 'freiburg is green',
                         2: 'the best dna is the dna of dinasours', 3: 'truth is absolute',
                         4: 'berlin is not that green', 5: 'truth manifests itself',
                         6: 'plato said truth is beautiful', 7: 'freiburg has dna'}

            vocab = set(itertools.chain(*list(map(lambda ab: ab.split(' '), abstracts.values()))))
            w2i = dict(zip(vocab, range(len(vocab))))
            word_to_count = [(w2i[word], sum(abstract.split(' ').count(word)
                                             for doc_id, abstract in abstracts.items())) for word in vocab]
            article_to_word = list(set([(doc_id, w2i[word])
                                        for doc_id, abstract in abstracts.items() for word in abstract.split(' ')]))
            article_to_word_to_count = list(set([(doc_id, w2i[word], abstract.count(word))
                                                 for doc_id, abstract in abstracts.items()
                                                 for word in abstract.split(' ')]))
            self.abstracts_preprocessor = AbstractsPreprocessor(abstracts, word_to_count,
                                                                article_to_word, article_to_word_to_count)
            self.documents, self.users = 8, 10
            self.ratings = numpy.array([[int(not bool((article + user) % 3))
                                         for article in range(self.documents)]
                                        for user in range(self.users)])

        self.verbose = verbose
        self.load_matrices = load_matrices
        self.dump = dump
        self.train_more = train_more
        self.random_seed = random_seed
        self.evaluator = Evaluator(self.ratings, self.abstracts_preprocessor, self.random_seed, self.verbose)
        self.config = RecommenderConfiguration()
        self.hyperparameters = self.config.get_hyperparameters()
        self.options = self.config.get_options()
        self.initializer = ModelInitializer(self.hyperparameters.copy(), self.options['n_iterations'], self.verbose)

    def run_lda(self):
        """
        Run LDA recommender.
        """
        lda_recommender = LDARecommender(self.initializer, self.evaluator, self.hyperparameters, self.options,
                                         self.verbose, self.load_matrices, self.dump)
        results = lda_recommender.train()
        report_str = 'Test sum {:.2f}, Train sum {:.2f}, Final error {:.5f}, train recall {:.5f}, '\
                     'test recall {:.5f}, recall@200 {:.5f}, '\
                     'ratio {:.5f}, mrr@5 {:.5f}, '\
                     'ndcg@5 {:.5f}, mrr@10 {:.5f}, ndcg@10 {:.5f}'
        print(report_str.format(*results))

    def run_lda2vec(self):
        """
        Runs LDA2Vec recommender.
        """
        lda2vec_recommender = LDA2VecRecommender(self.initializer, self.evaluator, self.hyperparameters,
                                                 self.options, self.verbose, self.load_matrices, self.dump)
        results = lda2vec_recommender.train()
        report_str = 'Test sum {:.2f}, Train sum {:.2f}, Final error {:.5f}, train recall {:.5f}, '\
                     'test recall {:.5f}, recall@200 {:.5f}, '\
                     'ratio {:.5f}, mrr@5 {:.5f}, '\
                     'ndcg@5 {:.5f}, mrr@10 {:.5f}, ndcg@10 {:.5f}'
        print(report_str.format(*results))

    def run_sdae(self):
        """
        Runs SDAE recommender.
        """
        sdae_recommender = SDAERecommender(self.initializer, self.evaluator, self.hyperparameters,
                                           self.options, self.verbose, self.load_matrices, self.dump)
        results = sdae_recommender.train()
        report_str = 'Test sum {:.2f}, Train sum {:.2f}, Final error {:.5f}, train recall {:.5f}, '\
                     'test recall {:.5f}, recall@200 {:.5f}, '\
                     'ratio {:.5f}, mrr@5 {:.5f}, '\
                     'ndcg@5 {:.5f}, mrr@10 {:.5f}, ndcg@10 {:.5f}'
        print(report_str.format(*results))

    def run_collaborative(self):
        """
        Runs collaborative filtering
        """
        ALS = CollaborativeFiltering(self.initializer, self.evaluator, self.hyperparameters, self.options,
                                     self.verbose, self.load_matrices, self.dump, self.train_more)

        results = ALS.train()
        report_str = 'Test sum {:.2f}, Train sum {:.2f}, Final error {:.5f}, train recall {:.5f}, '\
                     'test recall {:.5f}, recall@200 {:.5f}, '\
                     'ratio {:.5f}, mrr@5 {:.5f}, '\
                     'ndcg@5 {:.5f}, mrr@10 {:.5f}, ndcg@10 {:.5f}'
        print(report_str.format(*results))

    def run_grid_search(self):
        """
        Runs grid search
        """
        hyperparameters = {
            '_lambda': [0.01],
            'n_factors': [50, 100, 150, 200, 250, 300]
        }
        recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor, ratings=self.ratings,
                                        verbose=self.verbose, load_matrices=self.load_matrices,
                                        dump_matrices=self.dump, train_more=self.train_more,
                                        random_seed=self.random_seed)
        GS = GridSearch(recommender, hyperparameters, self.verbose)
        best_params, all_results = GS.train()

    def run_recommender(self):
        """
        Runs recommender
        """
        recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor, ratings=self.ratings,
                                        verbose=self.verbose, load_matrices=self.load_matrices,
                                        dump_matrices=self.dump, train_more=self.train_more,
                                        random_seed=self.random_seed)
        results = recommender.train()
        report_str = 'Test sum {:.2f}, Train sum {:.2f}, Final error {:.5f}, train recall {:.5f}, '\
                     'test recall {:.5f}, recall@200 {:.5f}, '\
                     'ratio {:.5f}, mrr@5 {:.5f}, '\
                     'ndcg@5 {:.5f}, mrr@10 {:.5f}, ndcg@10 {:.5f}'
        print(report_str.format(*results))
        recommender.dump_recommendations(200)

    def run_experiment(self):
        """
        Runs experiment
        """
        all_results = [['n_factors', '_lambda', 'desc', 'rmse', 'train_recall', 'test_recall', 'recall_at_200',
                        'ratio', 'mrr @ 5', 'ndcg @ 5', 'mrr @ 10', 'ndcg @ 10']]
        runs = RunsLoader()
        for run_idx, config_dict in enumerate(runs.get_runnable_recommenders()):
            if run_idx:
                print("\n___________________________________________________________________________________________")
            this_config = config_dict.copy()
            recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor, ratings=self.ratings,
                                            config=this_config, verbose=self.verbose, load_matrices=self.load_matrices,
                                            dump_matrices=self.dump, train_more=self.train_more,
                                            random_seed=self.random_seed)
            print("Run #%d %s: " % ((run_idx + 1), recommender.config.get_description()),
                  recommender.content_based, recommender.collaborative_filtering,
                  ", with: ", recommender.config.config_dict)
            recommender.train()
            current_result = [recommender.hyperparameters['n_factors'], recommender.hyperparameters['_lambda'],
                              recommender.config.get_description()]
            current_result.extend(recommender.get_evaluation_report())
            all_results.append(current_result)
        GridSearch(recommender, {}, self.verbose, report_name='experiment_results').dump_csv(all_results)

    def run_experiment_with_gridsearch(self):
        """
        Runs experiment after running grid search.
        """
        print("Getting Userbased hyperparameters")
        userbased_configs = {
            '_lambda': [0.01],
            'n_factors': [50, 100, 150, 200, 250, 300]
        }
        self.config.set_recommender_type('userbased')
        self.config.set_iterations(5)
        self.config.set_folds_num(1)
        recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor, ratings=self.ratings,
                                        config=self.config.get_all_config(), verbose=self.verbose,
                                        load_matrices=self.load_matrices, dump_matrices=False,
                                        train_more=self.train_more, random_seed=self.random_seed)
        userbased_hyperparameters, userbased_gridsearch_results =\
            GridSearch(recommender, userbased_configs, self.verbose, report_name='grid_search_userbased').train()

        print("Userbased hyperparameters:", userbased_hyperparameters)

        print("Getting Itembased hyperparameters")
        itembased_configs = {
            '_lambda': [0.01],
            'n_factors': [50, 100, 150, 200, 250, 300]
        }
        self.config.set_recommender_type('itembased')
        recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor, ratings=self.ratings,
                                        config=self.config.get_all_config(), verbose=self.verbose,
                                        load_matrices=self.load_matrices, dump_matrices=False,
                                        train_more=self.train_more, random_seed=self.random_seed)
        itembased_hyperparameters, itembased_gridsearch_results =\
            GridSearch(recommender, itembased_configs, self.verbose, report_name='grid_search_itembased').train()

        print("Itembased hyperparameters:", itembased_hyperparameters)

        for _ in range(5):
            print('.')
        print('Grid search done...')
        print('')
        print("Userbased hyperparameters:", userbased_hyperparameters)
        print("Itembased hyperparameters:", itembased_hyperparameters)

        all_results = [['n_factors', '_lambda', 'desc', 'rmse', 'train_recall', 'test_recall', 'recall_at_200',
                        'ratio', 'mrr @ 5', 'ndcg @ 5', 'mrr @ 10', 'ndcg @ 10']]
        runs = RunsLoader()
        for run_idx, config_dict in enumerate(runs.get_runnable_recommenders()):
            if run_idx:
                print("\n___________________________________________________________________________________________")
            this_config = config_dict.copy()
            if this_config['recommender']['recommender'] == 'itembased':
                if itembased_hyperparameters:
                    this_config['recommender']['hyperparameters'] = itembased_hyperparameters.copy()
            elif this_config['recommender']['recommender'] == 'userbased':
                if userbased_hyperparameters:
                    this_config['recommender']['hyperparameters'] = userbased_hyperparameters.copy()
            recommender = RecommenderSystem(abstracts_preprocessor=self.abstracts_preprocessor, ratings=self.ratings,
                                            config=this_config, verbose=self.verbose, load_matrices=self.load_matrices,
                                            dump_matrices=self.dump, train_more=self.train_more,
                                            random_seed=self.random_seed)
            print("Run #%d %s: " % ((run_idx + 1), recommender.config.get_description()),
                  recommender.content_based, recommender.collaborative_filtering,
                  ", with: ", recommender.config.config_dict)
            recommender.train()
            current_result = [recommender.hyperparameters['n_factors'], recommender.hyperparameters['_lambda'],
                              recommender.config.get_description()]
            current_result.extend(recommender.get_evaluation_report())
            all_results.append(current_result)
        GridSearch(recommender, {}, self.verbose, report_name='experiment_results').dump_csv(all_results)


if __name__ == '__main__':
    parser = OptionParser("runnables.py [options] [recommenders]\n\nRecommenders:\n\trecommender\n\tcollaborative"
                          "\n\tgrid_search\n\tlda\n\tlda2vec\n\tsdae\n\texperiment\n\texperiment_with_gridsearch")
    parser.add_option("-d", "--use-database", dest="db", action='store_true',
                      help="use database to run the recommender", metavar="DB")
    parser.add_option("-a", "--all", dest="all", action='store_true',
                      help="run every method", metavar="ALL")
    parser.add_option("-s", "--save", dest="dump", action='store_true',
                      help="dump the saved data into files in matrices/", metavar="DUMP")
    parser.add_option("-l", "--load", dest="load", action='store_true',
                      help="load saved models from files in matrices/", metavar="LOAD")
    parser.add_option("-v", "--verbose", dest="verbose", action='store_true',
                      help="print update statements during computations", metavar="VERBOSE")
    parser.add_option("-t", "--train_more", dest="train_more", action='store_true',
                      help="train the collaborative filtering more, after loading matrices", metavar="TRAINMORE")
    parser.add_option("-r", "--random_seed", dest="random_seed", action='store_true',
                      help="Set the seed to the current timestamp if true.", metavar="RANDOMSEED")
    options, args = parser.parse_args()
    use_database = options.db is not None
    use_all = options.all is not None
    load_matrices = options.load is not None
    verbose = options.verbose is not None
    dump = options.dump is not None
    train_more = options.train_more is not None
    random_seed = options.random_seed is not None

    if random_seed is True:
        numpy.random.seed(int(time.time()))
    runnable = RunnableRecommenders(use_database, verbose, load_matrices, dump, train_more, random_seed)
    if use_all is True:
        runnable.run_recommender()
        runnable.run_collaborative()
        runnable.run_grid_search()
        runnable.run_lda()
        runnable.run_lda2vec()
        runnable.run_sdae()
        runnable.run_experiment()
        runnable.run_experiment_with_gridsearch()
        sys.exit(0)
    found_runnable = False
    for arg in args:
        if arg == 'recommender':
            runnable.run_recommender()
            found_runnable = True
        elif arg == 'collaborative':
            runnable.run_collaborative()
            found_runnable = True
        elif arg == 'grid_search':
            runnable.run_grid_search()
            found_runnable = True
        elif arg == 'lda':
            runnable.run_lda()
            found_runnable = True
        elif arg == 'lda2vec':
            runnable.run_lda2vec()
            found_runnable = True
        elif arg == 'experiment':
            runnable.run_experiment()
            found_runnable = True
        elif arg == 'sdae':
            runnable.run_sdae()
            found_runnable = True
        elif arg == 'experiment_with_gridsearch':
            runnable.run_experiment_with_gridsearch()
            found_runnable = True
        else:
            print("'%s' option is not valid, please use one of "
                  "['recommender', 'collaborative', 'grid_search', 'lda', 'lda2vec', 'experiment', "
                  "'sdae', 'experiment_with_gridsearch']" % arg)
    if found_runnable is False:
        print("Didn't find any valid option, running recommender instead.")
        runnable.run_recommender()
