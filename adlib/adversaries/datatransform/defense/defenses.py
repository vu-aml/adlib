def huberreg(x, y, epss):
    scores = []
    bestclf, besteps, bestscore = None, None, 0
    for eps in epss:
        clf = lm.HuberRegressor(epsilon=eps, max_iter=10000, alpha=1e-5)
        clf.fit(x, y)
        score = clf.score(x[~clf.outliers_], y[~clf.outliers_])
        scores.append(score)
        if score > bestscore:
            bestclf, besteps, bestscore = clf, eps, score
    return bestclf, besteps, scores


def ransacmodel(x, y, model, lam, counts):
    allmodels = {'lasso': lm.Lasso(alpha=lam, max_iter=10000),
                 'ridge': lm.Ridge(alpha=lam),
                 'enet': lm.ElasticNet(alpha=lam, max_iter=10000),
                 'linreg': lm.Ridge(alpha=0.00001)}
    scores = []
    bestclf, bestcount, bestscore = None, None, 0
    for count in counts:
        clfransac = lm.RANSACRegressor(allmodels[model],
                                       min_samples=count)  # allmodels[model])
        clfransac.fit(x, y)
        score = clfransac.score(x[clfransac.inlier_mask_],
                                y[clfransac.inlier_mask_])
        scores.append(score)
        if score > bestscore:
            bestclf, bestcount, bestscore = clfransac, count, score
    return bestclf, bestcount, scores


def trimclf(x, y, count, lam, model):
    length = x.shape[0]
    width = x.shape[1]
    y = np.array(y)
    inds = sorted(np.random.permutation(length))[:count]
    initinds = inds[:]
    clf = None

    newinds = []
    it = 0
    toterr = 10000
    lasterr = 20000
    clf, _ = learnmodel(x, y, model, lam)

    while (sorted(inds) != sorted(
            newinds) and it < 400 and lasterr - toterr > 1e-5):
        newinds = inds[:]
        lasterr = toterr
        subx = x[inds]
        suby = y[inds]
        clf.fit(subx, suby)
        preds = clf.predict(x) - y
        residvec = np.square(preds)

        residtopns = sorted([(residvec[i], i) for i in range(length)])[:count]
        resid = [val[1] for val in residtopns]
        topnresid = [val[0] for val in residtopns]

        # set inds to indices of n largest values in error
        inds = sorted(resid)
        # recompute error
        toterr = sum(topnresid)
        it += 1
    return clf, lam, inds


def tip(a, b, n):
    a = np.reshape(a, (a.shape[0], 1))
    b = np.reshape(b, (a.shape[0], 1))
    prods = [a[i] * b[i] for i in range(a.shape[0])]
    sortprods = sorted([(abs(j), i) for i, j in enumerate(prods)])
    inds = [i for j, i in sortprods][:n]
    correl = sum([prods[i] for i in inds])
    selfcorr = sum([b[i] * b[i] for i in inds])
    return correl / selfcorr if correl != 0 else 0


def chenrotr(x, y, count, ks):
    print(x.shape, y.shape, count, ks)
    corrs = [tip(y, col, count) for col in x.T]
    bestcorrs = sorted([(abs(j), i) for i, j in enumerate(corrs)], reverse=True)
    bestmodel, besterrs, bestk = None, -1, 0
    allerrs = []
    for k in ks:
        inds = [i for j, i in bestcorrs[:k]]
        model = np.zeros(x.shape[1])
        for i in inds:
            model[i] = corrs[i]
        chp = chenpred(model, x)
        curres = [(chp[i] - y[i]) ** 2 for i in range(x.shape[0])]
        curerr = sum(curres) / len(curres)
        allerrs.append(curerr)
        if besterrs != -1:
            if curerr < besterrs:
                bestmodel = model
                besterrs = curerr
                bestk = k
        else:
            bestmodel = model
            besterrs = curerr
            bestk = k
    return np.array(bestmodel), bestk, allerrs


def chenpred(chmod, x):
    return np.dot(x, chmod)


def ronidefense(x, y, count, lam, model, trainsizes):
    allerrs = []
    bestclf, besttsize, bestcleans, bestscore = None, None, [], 0
    for trainsize in trainsizes:
        print(trainsize)
        all_increases = []
        for i in range(x.shape[0]):
            curx, cury = x[i], y[i]
            cur_increase = 0
            for j in range(RONI_TRIALS):
                # sample train, valid set
                traininds = np.random.choice(x.shape[0], size=trainsize)
                validinds = np.random.choice(x.shape[0], size=RONI_VALID_SIZE)

                trainx, trainy = x[traininds], y[traininds]
                validx, validy = x[validinds], y[validinds]

                # train on both train and train plus point
                exclude, _ = learnmodel(trainx, trainy, model, lam)
                include, _ = learnmodel(
                    np.append(trainx, curx.reshape((1, -1)), axis=0),
                    np.append(trainy, cury.reshape((1,)), axis=0), model, lam)

                exclude_valid = exclude.predict(validx)
                exclude_mse = mean_squared_error(exclude_valid, validy)

                include_valid = include.predict(validx)
                include_mse = mean_squared_error(include_valid, validy)

                cur_increase += (include_mse - exclude_mse)

            all_increases.append(cur_increase)

        # get smallest <count> increases and retrain
        sort_increases = sorted(enumerate(all_increases),
                                key=lambda tup: tup[1])
        clean_inds = [x[0] for x in sort_increases[:count]]
        clean_x, clean_y = x[clean_inds], y[clean_inds]
        clean_model, clean_lam = learnmodel(clean_x, clean_y, model, lam)
        score = clean_model.score(clean_x, clean_y)
        allerrs.append(score)
        if score > bestscore:
            bestclf, besttsize, bestcleaninds, bestscore = clean_model, trainsize, clean_inds, score
    return clean_model, lam, bestcleaninds, allerrs, besttsize
