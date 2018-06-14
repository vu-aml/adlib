# Matthew Jagielski

import numpy as np, scipy.optimize
import numpy.linalg as la

# import my modules
from adlib.adversaries.datatransform.poisoning.my_args import setup_argparse
from adlib.adversaries.datatransform.poisoning.gd_poisoners import *


# -------------------------------------------------------------------------------
# TRIM algorithm
# ------------------------------------------------------------------------------- 
def robustopt(x, y, count, lam, poiser):
    length = x.shape[0]
    width = x.shape[1]
    y = np.array(y)
    tau = sorted(np.random.permutation(length))[:count]
    inittau = tau[:]
    clf = None

    newtau = []
    it = 0
    toterr = 10000
    lasterr = 20000

    clf, _ = poiser.learn_model(x, y, None)

    while (sorted(tau) != sorted(
            newtau) and it < 400 and lasterr - toterr > 1e-5):
        newtau = tau[:]
        lasterr = toterr
        subx = x[tau]
        suby = y[tau]
        clf.fit(subx, suby)
        w, b = clf.coef_, clf.intercept_

        residvec = [(w * np.transpose(x[i]) + b - y[i]) ** 2 for i in
                    range(length)]

        residtopns = sorted([(residvec[i], i) for i in range(length)])[:count]
        resid = [val[1] for val in residtopns]
        topnresid = [val[0] for val in residtopns]

        # set tau to indices of n largest values in error
        tau = sorted(resid)  # [1 if i in resid else 0 for i in range(length)]
        # recompute error
        toterr = sum(topnresid)
        it += 1
    return clf, w, b, lam, tau


def open_dataset(f, visualize):
    if visualize:
        rng = np.random.RandomState(1)
        random_state = 1
        x, y = make_regression(n_samples=300, n_features=1,
                               random_state=random_state, noise=15.0, bias=1.5)
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        plt.plot(x, y, 'k.')
        colmap = []
    else:
        x, y = read_dataset_file(f)

    return np.matrix(x), y


def read_dataset_file(f):
    with open(f) as dataset:
        x = []
        y = []
        cols = dataset.readline().split(',')
        print(cols)

        global colmap
        colmap = {}
        for i, col in enumerate(cols):
            if ':' in col:
                if col.split(':')[0] in colmap:
                    colmap[col.split(':')[0]].append(i - 1)
                else:
                    colmap[col.split(':')[0]] = [i - 1]
        for line in dataset:
            line = [float(val) for val in line.split(',')]
            y.append(line[0])
            x.append(line[1:])

        return np.matrix(x), y


# -------------------------------------------------------------------------------
def open_logging_files(logdir, modeltype, logind, args):
    myname = str(modeltype) + str(logind)
    logdir = logdir + os.path.sep + myname
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, 'cmd'), 'w') as cmdfile:
        cmdfile.write(' '.join(['python3'] + argv))
        cmdfile.write('\n')
        for arg in args.__dict__:
            cmdfile.write('{}: {}\n'.format(arg, args.__dict__[arg]))

    trainfile = open(logdir + os.path.sep + "train.txt", 'w')
    testfile = open(logdir + os.path.sep + "test.txt", 'w')
    validfile = open(logdir + os.path.sep + "valid.txt", 'w')
    resfile = open(logdir + os.path.sep + "err.txt", 'w')
    resfile.write('poisct,itnum,obj_diff,obj_val,val_mse,test_mse,time\n')
    return trainfile, testfile, validfile, resfile, logdir


# ------------------------------------------------------------------------------- 
def sample_dataset(x, y, trnct, poisct, tstct, vldct, seed):
    size = x.shape[0]
    print(size)

    np.random.seed(seed)
    fullperm = np.random.permutation(size)

    sampletrn = fullperm[:trnct]
    sampletst = fullperm[trnct:trnct + tstct]
    samplevld = fullperm[trnct + tstct:trnct + tstct + vldct]
    samplepois = np.random.choice(size, poisct)

    trnx = np.matrix(
        [np.array(x[row]).reshape((x.shape[1],)) for row in sampletrn])
    trny = [y[row] for row in sampletrn]

    tstx = np.matrix(
        [np.array(x[row]).reshape((x.shape[1],)) for row in sampletst])
    tsty = [y[row] for row in sampletst]

    poisx = np.matrix(
        [np.array(x[row]).reshape((x.shape[1],)) for row in samplepois])
    poisy = [y[row] for row in samplepois]

    vldx = np.matrix(
        [np.array(x[row]).reshape((x.shape[1],)) for row in samplevld])
    vldy = [y[row] for row in samplevld]

    return trnx, trny, tstx, tsty, poisx, poisy, vldx, vldy


# -------------------------------------------------------------------------------
def infflip(x, y, count, poiser):
    mean = np.ravel(x.mean(axis=0))  # .reshape(1,-1)
    corr = np.dot(x.T, x) + 0.01 * np.eye(x.shape[1])
    invmat = np.linalg.pinv(corr)
    hmat = x * invmat * np.transpose(x)
    allgains = []
    for i in range(x.shape[0]):
        posgain = (np.sum(hmat[i]) * (1 - y[i]), 1)
        neggain = (np.sum(hmat[i]) * y[i], 0)
        allgains.append(max(posgain, neggain))

    totalprob = sum([a[0] for a in allgains])
    allprobs = [0]
    for i in range(len(allgains)):
        allprobs.append(allprobs[-1] + allgains[i][0])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))
    gainsy = [allgains[ind][1] for ind in poisinds]

    # sortedgains = sorted(enumerate(allgains),key = lambda tup: tup[1])[:count]
    # poisinds = [a[0] for a in sortedgains]
    # bestgains = [a[1][1] for a in sortedgains]

    return x[poisinds], gainsy


# -------------------------------------------------------------------------------
def levflip(x, y, count, poiser):
    allpoisy = []
    clf, _ = poiser.learn_model(x, y, None)
    mean = np.ravel(x.mean(axis=0))  # .reshape(1,-1)
    corr = np.dot(x.T, x) + 0.01 * np.eye(x.shape[1])
    invmat = np.linalg.pinv(corr)
    hmat = x * invmat * np.transpose(x)

    alllevs = [hmat[i, i] for i in range(x.shape[0])]
    totalprob = sum(alllevs)
    allprobs = [0]
    for i in range(len(alllevs)):
        allprobs.append(allprobs[-1] + alllevs[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        curind = bisect.bisect_left(allprobs, a)
        poisinds.append(curind)
        if clf.predict(x[curind].reshape(1, -1)) < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy


# -------------------------------------------------------------------------------
def cookflip(x, y, count, poiser):
    allpoisy = []
    clf, _ = poiser.learn_model(x, y, None)
    preds = [clf.predict(x[i].reshape(1, -1)) for i in range(x.shape[0])]
    errs = [(y[i] - preds[i]) ** 2 for i in range(x.shape[0])]
    mean = np.ravel(x.mean(axis=0))  # .reshape(1,-1)
    corr = np.dot(x.T, x) + 0.01 * np.eye(x.shape[1])
    invmat = np.linalg.pinv(corr)
    hmat = x * invmat * np.transpose(x)

    allcooks = [hmat[i, i] * errs[i] / (1 - hmat[i, i]) ** 2 for i in
                range(x.shape[0])]

    totalprob = sum(allcooks)

    allprobs = [0]
    for i in range(len(allcooks)):
        allprobs.append(allprobs[-1] + allcooks[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        curind = bisect.bisect_left(allprobs, a)
        poisinds.append(curind)
        if clf.predict(x[curind].reshape(1, -1)) < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy


# -------------------------------------------------------------------------------
def farthestfirst(x, y, count, poiser):
    allpoisy = []
    clf, _ = poiser.learn_model(x, y, None)
    preds = [clf.predict(x[i].reshape(1, -1)) for i in range(x.shape[0])]
    errs = [(y[i] - preds[i]) ** 2 for i in range(x.shape[0])]
    totalprob = sum(errs)
    allprobs = [0]
    for i in range(len(errs)):
        allprobs.append(allprobs[-1] + errs[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        curind = bisect.bisect_left(allprobs, a)
        poisinds.append(curind)
        if preds[curind] < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy


# -------------------------------------------------------------------------------
def alfatilt(x, y, count, poiser):
    trueclf, _ = poiser.learn_model(x, y, None)
    truepreds = trueclf.predict(x)

    goalmodel = np.random.uniform(low=-1 / sqrt(x.shape[1]),
                                  high=1 / sqrt(x.shape[1]),
                                  shape=(x.shape[1] + 1))
    goalpreds = np.dot(x, goalmodel[:-1]) + goalmodel[-1].item()

    svals = np.square(trueclf.predict(x) - y)  # squared error
    svals = svals / svals.max()
    qvals = np.square(goalpreds - y)
    qvals = qvals / qvals.max()

    flipscores = (svals + qvals).tolist()

    totalprob = sum(flipscores)
    allprobs = [0]
    allpoisy = []
    for i in range(len(flipscores)):
        allprobs.append(allprobs[-1] + flipscores[i])
    allprobs = allprobs[1:]
    poisinds = []
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))
        if truepreds[curind] < 0.5:
            allpoisy.append(1)
        else:
            allpoisy.append(0)

    return x[poisinds], allpoisy


# -------------------------------------------------------------------------------
def inf_flip(X_tr, Y_tr, count):
    Y_tr = np.array(Y_tr)
    inv_cov = (0.01 * np.eye(X_tr.shape[1]) + np.dot(X_tr.T, X_tr)) ** -1
    H = np.dot(np.dot(X_tr, inv_cov), X_tr.T)
    bests = np.sum(H, axis=1)
    room = .5 + np.abs(Y_tr - 0.5)
    yvals = 1 - np.floor(0.5 + Y_tr)
    stat = np.multiply(bests.ravel(), room.ravel())
    stat = stat.tolist()[0]
    totalprob = sum(stat)
    allprobs = [0]
    poisinds = []
    for i in range(X_tr.shape[0]):
        allprobs.append(allprobs[-1] + stat[i])
    allprobs = allprobs[1:]
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))

    return X_tr[poisinds], [yvals[a] for a in poisinds]


# -------------------------------------------------------------------------------
def alfa_tilt(X_tr, Y_tr, count):
    inv_cov = (0.01 * np.eye(X_tr.shape[1]) + np.dot(X_tr.T, X_tr)) ** -1
    H = np.dot(np.dot(X_tr, inv_cov), X_tr.T)
    randplane = np.random.standard_normal(size=X_tr.shape[1] + 1)
    w, b = randplane[:-1], randplane[-1]
    preds = np.dot(X_tr, w) + b
    yvals = preds.clip(0, 1)
    yvals = 1 - np.floor(0.5 + yvals)
    diff = yvals - Y_tr
    print(diff)
    yvals = yvals.tolist()[0]
    changes = np.dot(diff, H).tolist()[0]
    changes = [max(a, 0) for a in changes]

    totalprob = sum(changes)
    allprobs = [0]
    poisinds = []
    for i in range(X_tr.shape[0]):
        allprobs.append(allprobs[-1] + changes[i])
    allprobs = allprobs[1:]
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))
    return X_tr[poisinds], [yvals[a] for a in poisinds]


# -------------------------------------------------------------------------------
def adaptive(X_tr, Y_tr, count):
    Y_tr_copy = np.array(Y_tr)
    X_tr_copy = np.copy(X_tr)
    print(np.allclose(X_tr_copy, X_tr))
    room = .5 + np.abs(Y_tr_copy)
    yvals = 1 - np.floor(0.5 + Y_tr_copy)
    diff = (yvals - Y_tr_copy).ravel()
    poisinds = []
    X_pois = np.zeros((count, X_tr.shape[1]))
    Y_pois = []
    for i in range(count):
        print(X_tr_copy.shape, diff.shape)
        inv_cov = np.linalg.inv(
            0.01 * np.eye(X_tr_copy.shape[1]) + np.dot(X_tr_copy.T, X_tr_copy))
        H = np.dot(np.dot(X_tr_copy, inv_cov), X_tr_copy.T)
        bests = np.sum(H, axis=1)
        stat = np.multiply(bests.ravel(), diff)
        # indtoadd = np.argmax(stat)
        indtoadd = np.random.choice(stat.shape[0],
                                    p=np.abs(stat) / np.sum(np.abs(stat)))
        print(indtoadd)
        X_pois[i] = X_tr_copy[indtoadd, :]
        X_tr_copy = np.delete(X_tr_copy, indtoadd, axis=0)
        diff = np.delete(diff, indtoadd, axis=0)
        Y_pois.append(yvals[indtoadd])
        yvals = np.delete(yvals, indtoadd, axis=0)
    print(X_pois)
    print(Y_pois)
    return np.matrix(X_pois), Y_pois


def randflip(X_tr, Y_tr, count):
    poisinds = np.random.choice(X_tr.shape[0], count, replace=False)
    print("Points selected: ", poisinds)
    # Y_pois = [1-Y_tr[i] for i in poisinds]  # this is for validating yopt, not for initialization
    Y_pois = [1 if 1 - Y_tr[i] > 0.5 else 0 for i in
              poisinds]  # this is the flip all the way implementation
    return np.matrix(X_tr[poisinds]), Y_pois


def randflipnobd(X_tr, Y_tr, count):
    poisinds = np.random.choice(X_tr.shape[0], count, replace=False)
    print("Points selected: ", poisinds)
    Y_pois = [1 - Y_tr[i] for i in
              poisinds]  # this is for validating yopt, not for initialization
    # Y_pois = [1 if 1-Y_tr[i]>0.5 else 0 for i in poisinds]  # this is the flip all the way implementation
    return np.matrix(X_tr[poisinds]), Y_pois


def rmml(X_tr, Y_tr, count):
    print(X_tr.shape, len(Y_tr), count)
    mean = np.ravel(X_tr.mean(axis=0))  # .reshape(1,-1)
    covar = np.dot((X_tr - mean).T, (X_tr - mean)) / X_tr.shape[
        0] + 0.01 * np.eye(X_tr.shape[1])
    model = linear_model.Ridge(alpha=.01)
    model.fit(X_tr, Y_tr)
    allpoisx = np.random.multivariate_normal(mean, covar, size=count)
    allpoisx[allpoisx >= 0.5] = 1
    allpoisx[allpoisx < 0.5] = 0
    poisy = model.predict(allpoisx)
    poisy = 1 - poisy
    poisy[poisy >= 0.5] = 1
    poisy[poisy < 0.5] = 0
    print(allpoisx.shape, poisy.shape)
    for i in range(count):
        curpoisxelem = allpoisx[i, :]
        for col in colmap:
            vals = [(curpoisxelem[j], j) for j in colmap[col]]
            topval, topcol = max(vals)
            for j in colmap[col]:
                if j != topcol:
                    curpoisxelem[j] = 0
            if topval > 1 / (1 + len(colmap[col])):
                curpoisxelem[topcol] = 1
            else:
                curpoisxelem[topcol] = 0
        allpoisx[i] = curpoisxelem
    return np.matrix(allpoisx), poisy.tolist()


# -------------------------------------------------------------------------------
def roundpois(poisx, poisy):
    return np.around(poisx), [0 if val < 0.5 else 1 for val in poisy]


# ------------------------------------------------------------------------------- 
# #datasets = ["icmldataset.txt",'contagio-preprocessed-missing.csv','pharm-preproc.csv','loan-processed.csv','house-processed.csv']
# ------------------------------------------------------------------------------- 
def main(args):
    trainfile, testfile, validfile, resfile, newlogdir = \
        open_logging_files(args.logdir, args.model, args.logind, args)
    x, y = open_dataset(args.dataset, args.visualize)
    trainx, trainy, testx, testy, poisx, poisy, validx, validy = \
        sample_dataset(x, y, args.trainct, args.poisct, args.testct,
                       args.validct, \
                       args.seed)

    for i in range(len(testy)):
        testfile.write(','.join(
            [str(val) for val in [testy[i]] + testx[i].tolist()[0]]) + '\n')
    testfile.close()

    for i in range(len(validy)):
        validfile.write(','.join(
            [str(val) for val in [validy[i]] + validx[i].tolist()[0]]) + '\n')
    validfile.close()

    for i in range(len(trainy)):
        trainfile.write(','.join(
            [str(val) for val in [trainy[i]] + trainx[i].tolist()[0]]) + '\n')

    print(la.matrix_rank(trainx))
    print(trainx.shape)

    totprop = args.poisct / (args.poisct + args.trainct)
    print(totprop)

    timestart, timeend = None, None
    types = {'linreg': LinRegGDPoisoner, \
             'lasso': LassoGDPoisoner, \
             'enet': ENetGDPoisoner, \
             'ridge': RidgeGDPoisoner}

    inits = {'levflip': levflip, \
             'cookflip': cookflip, \
             'alfatilt': alfa_tilt, \
             'inflip': inf_flip, \
             'ffirst': farthestfirst, \
             'adaptive': adaptive, \
             'randflip': randflip, \
             'randflipnobd': randflipnobd, \
             'rmml': rmml}

    bestpoisx, bestpoisy, besterr = None, None, -1

    init = inits[args.initialization]

    genpoiser = types[args.model](trainx, trainy, testx, testy, validx, validy,
                                  args.eta, args.beta, args.sigma, args.epsilon,
                                  args.multiproc,
                                  trainfile, resfile, args.objective,
                                  args.optimizey, colmap)

    for initit in range(args.numinit):
        poisx, poisy = init(trainx, trainy,
                            int(args.trainct * totprop / (1 - totprop) + 0.5))
        clf, _ = genpoiser.learn_model(np.concatenate((trainx, poisx), axis=0),
                                       trainy + poisy, None)
        err = genpoiser.computeError(clf)[0]
        print("Validation Error:", err)
        if err > besterr:
            bestpoisx, bestpoisy, besterr = np.copy(poisx), poisy[:], err
    poisx, poisy = np.matrix(bestpoisx), bestpoisy
    poiser = types[args.model](trainx, trainy, testx, testy, validx, validy, \
                               args.eta, args.beta, args.sigma, args.epsilon, \
                               args.multiproc, trainfile, resfile, \
                               args.objective, args.optimizey, colmap)

    for i in range(args.partct + 1):
        curprop = (i + 1) * totprop / (args.partct + 1)
        numsamples = int(0.5 + args.trainct * (curprop / (1 - curprop)))
        curpoisx = poisx[:numsamples, :]
        curpoisy = poisy[:numsamples]
        trainfile.write("\n")

        timestart = datetime.datetime.now()
        poisres, poisresy = poiser.poison_data(curpoisx, curpoisy, timestart,
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
        resfile.write(','.join([str(val) for val in towrite]) + "\n")
        trainfile.write("\n")
        for j in range(numsamples):
            trainfile.write(','.join([str(val) for val in
                                      [poisresy[j]] + poisres[j].tolist()[
                                          0]]) + '\n')

        if args.rounding:
            towrite = [numsamples, 'r', None, None, rounderr[0], rounderr[1],
                       (timeend - timestart).total_seconds()]
            resfile.write(','.join([str(val) for val in towrite]) + "\n")
            trainfile.write("\nround\n")
            for j in range(numsamples):
                trainfile.write(','.join([str(val) for val in
                                          [roundy[j]] + roundx[j].tolist()[
                                              0]]) + '\n')

        resfile.flush()
        trainfile.flush()
        os.fsync(resfile.fileno())
        os.fsync(trainfile.fileno())

    trainfile.close()
    testfile.close()

    print()
    print("Unpoisoned")
    print("Validation MSE:", errgrd[0])
    print("Test MSE:", errgrd[1])
    print('Poisoned:')
    print("Validation MSE:", err[0])
    print("Test MSE:", err[1])
    if args.rounding:
        print("Rounded")
        print("Validation MSE", rounderr[0])
        print("Test MSE:", rounderr[1])


if __name__ == '__main__':
    print("starting poison ...\n")
    parser = setup_argparse()
    args = parser.parse_args()

    print("-----------------------------------------------------------")
    print(args)
    print("-----------------------------------------------------------")
    main(args)
