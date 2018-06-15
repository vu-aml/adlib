# data_transform.py
# A data-transformation implementation based on IEEE S&P 2018 paper
# 'Manipulating Machine Learning: Poisoning Attacks and Countermeasures for
# Regression Learning.'
# Matthew Sedam. 2018. Original code from Matthew Jagielski.

from adlib.adversaries import Adversary
from adlib.adversaries.datatransform.poisoning.poison import *
from typing import Dict
from argparse import Namespace


class DataTransform(Adversary):
    def __init__(self, beta=0.1,
                 dataset=('./data_reader/data/raw/data-transform/'
                          'house-processed.csv'),
                 epsilon=0.001, eta=0.5, initialization='randflip', lambd=1,
                 logdir='./results', logind=0, model='ridge', multiproc=False,
                 numinit=1, objective=1, optimizey=False, partct=4, poisct=75,
                 rounding=False, seed=123, sigma=1.0, testct=500, trainct=300,
                 validct=250, visualize=False):

        Adversary.__init__(self)
        self.beta = beta
        self.dataset = dataset
        self.epsilon = epsilon
        self.eta = eta
        self.initialization = initialization
        self.lambd = lambd
        self.logdir = logdir
        self.logind = logind
        self.model = model
        self.multiproc = multiproc
        self.numinit = numinit
        self.objective = objective
        self.optimizey = optimizey
        self.partct = partct
        self.poisct = poisct
        self.rounding = rounding
        self.seed = seed
        self.sigma = sigma
        self.testct = testct
        self.trainct = trainct
        self.validct = validct
        self.visualize = visualize
        self.args = self.get_available_params()

    def attack(self, instances):
        x = instances[0]
        y = instances[1]
        args = Namespace(**self.args)

        colmap = {}
        trainfile, testfile, validfile, resfile, newlogdir = \
            open_logging_files(args.logdir, args.model, args.logind, args)
        trainx, trainy, testx, testy, poisx, poisy, validx, validy = \
            sample_dataset(x, y, args.trainct, args.poisct, args.testct,
                           args.validct, args.seed)

        for i in range(len(testy)):
            testfile.write(','.join(
                [str(val) for val in [testy[i]] + testx[i].tolist()[0]]) + '\n')
        testfile.close()

        for i in range(len(validy)):
            validfile.write(','.join(
                [str(val) for val in
                 [validy[i]] + validx[i].tolist()[0]]) + '\n')
        validfile.close()

        for i in range(len(trainy)):
            trainfile.write(','.join(
                [str(val) for val in
                 [trainy[i]] + trainx[i].tolist()[0]]) + '\n')

        print(la.matrix_rank(trainx))
        print(trainx.shape)

        totprop = args.poisct / (args.poisct + args.trainct)
        print(totprop)

        timestart, timeend = None, None
        types = {'linreg': LinRegGDPoisoner,
                 'lasso': LassoGDPoisoner,
                 'enet': ENetGDPoisoner,
                 'ridge': RidgeGDPoisoner}

        inits = {'levflip': levflip,
                 'cookflip': cookflip,
                 'alfatilt': alfa_tilt,
                 'inflip': inf_flip,
                 'ffirst': farthestfirst,
                 'adaptive': adaptive,
                 'randflip': randflip,
                 'randflipnobd': randflipnobd,
                 'rmml': rmml}

        bestpoisx, bestpoisy, besterr = None, None, -1

        init = inits[args.initialization]

        genpoiser = types[args.model](trainx, trainy, testx, testy, validx,
                                      validy,
                                      args.eta, args.beta, args.sigma,
                                      args.epsilon,
                                      args.multiproc,
                                      trainfile, resfile, args.objective,
                                      args.optimizey, colmap)

        for initit in range(args.numinit):
            poisx, poisy = init(trainx, trainy,
                                int(args.trainct * totprop / (
                                        1 - totprop) + 0.5))
            clf, _ = genpoiser.learn_model(
                np.concatenate((trainx, poisx), axis=0),
                trainy + poisy, None)
            err = genpoiser.computeError(clf)[0]
            print('Validation Error:', err)
            if err > besterr:
                bestpoisx, bestpoisy, besterr = np.copy(poisx), poisy[:], err
        poisx, poisy = np.matrix(bestpoisx), bestpoisy
        poiser = types[args.model](trainx, trainy, testx, testy, validx, validy,
                                   args.eta, args.beta, args.sigma,
                                   args.epsilon,
                                   args.multiproc, trainfile, resfile,
                                   args.objective, args.optimizey, colmap)

        for i in range(args.partct + 1):
            curprop = (i + 1) * totprop / (args.partct + 1)
            numsamples = int(0.5 + args.trainct * (curprop / (1 - curprop)))
            curpoisx = poisx[:numsamples, :]
            curpoisy = poisy[:numsamples]
            trainfile.write('\n')

            timestart = datetime.datetime.now()
            poisres, poisresy = poiser.poison_data(curpoisx, curpoisy,
                                                   timestart,
                                                   args.visualize, newlogdir)
            print(poisres.shape, trainx.shape)
            poisedx = np.concatenate((trainx, poisres), axis=0)
            poisedy = trainy + poisresy

            clfp, _ = poiser.learn_model(poisedx, poisedy, None)
            clf = poiser.initclf
            if args.rounding:
                roundx, roundy = roundpois(poisres, poisresy)
                rpoisedx, rpoisedy = np.concatenate((trainx, roundx),
                                                    axis=0), trainy + roundy
                clfr, _ = poiser.learn_model(rpoisedx, rpoisedy, None)
                rounderr = poiser.computeError(clfr)

            errgrd = poiser.computeError(clf)
            err = poiser.computeError(clfp)

            timeend = datetime.datetime.now()

            towrite = [numsamples, -1, None, None, err[0], err[1],
                       (timeend - timestart).total_seconds()]
            resfile.write(','.join([str(val) for val in towrite]) + '\n')
            trainfile.write('\n')
            for j in range(numsamples):
                trainfile.write(','.join([str(val) for val in
                                          [poisresy[j]] +
                                          poisres[j].tolist()[0]]) + '\n')

            if args.rounding:
                towrite = [numsamples, 'r', None, None, rounderr[0],
                           rounderr[1],
                           (timeend - timestart).total_seconds()]
                resfile.write(','.join([str(val) for val in towrite]) + '\n')
                trainfile.write('\nround\n')
                for j in range(numsamples):
                    trainfile.write(','.join([str(val) for val in
                                              [roundy[j]] +
                                              roundx[j].tolist()[0]]) + '\n')

            resfile.flush()
            trainfile.flush()
            os.fsync(resfile.fileno())
            os.fsync(trainfile.fileno())

        trainfile.close()
        testfile.close()

        print()
        print('Unpoisoned')
        print('Validation MSE:', errgrd[0])
        print('Test MSE:', errgrd[1])
        print('Poisoned:')
        print('Validation MSE:', err[0])
        print('Test MSE:', err[1])
        if args.rounding:
            print('Rounded')
            print('Validation MSE', rounderr[0])
            print('Test MSE:', rounderr[1])

        return poisedx, poisedy

    def set_params(self, params: Dict):
        if params['beta'] is not None:
            self.beta = params['beta']
        if params['dataset'] is not None:
            self.dataset = params['dataset']
        if params['epsilon'] is not None:
            self.epsilon = params['epsilon']
        if params['eta'] is not None:
            self.eta = params['eta']
        if params['initialization'] is not None:
            self.initialization = params['initialization']
        if params['lambd'] is not None:
            self.lambd = params['lambd']
        if params['logdir'] is not None:
            self.logdir = params['logdir']
        if params['logind'] is not None:
            self.logind = params['logind']
        if params['model'] is not None:
            self.model = params['model']
        if params['multiproc'] is not None:
            self.multiproc = params['multiproc']
        if params['numinit'] is not None:
            self.numinit = params['numinit']
        if params['objective'] is not None:
            self.objective = params['objective']
        if params['optimizey'] is not None:
            self.optimizey = params['optimizey']
        if params['partct'] is not None:
            self.partct = params['partct']
        if params['poisct'] is not None:
            self.poisct = params['poisct']
        if params['rounding'] is not None:
            self.rounding = params['rounding']
        if params['seed'] is not None:
            self.seed = params['seed']
        if params['sigma'] is not None:
            self.sigma = params['sigma']
        if params['testsct'] is not None:
            self.testct = params['testct']
        if params['trainct'] is not None:
            self.trainct = params['trainct']
        if params['validct'] is not None:
            self.validct = params['validct']
        if params['visualize'] is not None:
            self.visualize = params['visualize']

    def get_available_params(self):
        params = {'beta': self.beta,
                  'dataset': self.dataset,
                  'epsilon': self.epsilon,
                  'eta': self.eta,
                  'initialization': self.initialization,
                  'lambd': self.lambd,
                  'logdir': self.logdir,
                  'logind': self.logind,
                  'model': self.model,
                  'multiproc': self.multiproc,
                  'numinit': self.numinit,
                  'objective': self.objective,
                  'optimizey': self.optimizey,
                  'partct': self.partct,
                  'poisct': self.poisct,
                  'rounding': self.rounding,
                  'seed': self.seed,
                  'sigma': self.sigma,
                  'testct': self.testct,
                  'trainct': self.trainct,
                  'validct': self.validct,
                  'visualize': self.visualize}

        return params

    def set_adversarial_params(self, learner, train_instances):
        pass
