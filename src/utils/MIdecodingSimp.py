import numpy as np
import matplotlib.pylab as plt
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def flatten(X):
    X = X.reshape((X.shape[0],-1))
    return X

def estimateMIS(
    data,
    overtime=True,
    n_bootstraps=25,
    ci=[0.25, 0.75],
    Crange=None,
    gammarange=None,
    verbose=False,
):
    """
    Estimates the mutual information between a collection of time series and a list
    of labels, where each label is associated with one set of time series and all
    labels are equally likely.
    Follows the method of Granados, Pietsch, et al,
    Proc Nat Acad Sci USA 115 (2008) 6088.
    Uses sklean to optimise a pipeline for classifying the individual time series,
    choosing the number of PCA components (3-7), the classifier - a support vector
    machine with either a linear or a radial basis function kernel - and its C and
    gamma parameters.
    Errors are found using bootstrapped datasets.
    Parameters
    ---------
    data:  list of arrays
            A list of arrays, where each array comprises a collection of time series
            from a different class and so the length of the list gives the number of
            classes. For the array for one class, each row is a single-cell time
            series.
    overtime: boolean (default: False)
            If True, calculate the mutual information as a function of the duration of
            the time series, by finding the mutuation information for all possible
            sub-time series that start from t= 0.
    n_bootstraps: int, optional
            The number of bootstraps used to estimate errors.
    ci: 1x2 array or list, optional
            The lower and upper confidence intervals.
            E.g. [0.25, 0.75] for the interquartile range
    Crange: array, optional
            An array of potential values for the C parameter of the support vector machine
            and from which the optimal value of C will be chosen.
            If None, np.logspace(-3, 3, 10) is used. This range should be increased if
            the optimal C is one of the boundary values.
            See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    gammarange: array, optional
            An array of potential values for the gamma parameter for the radial basis
            function kernel of the support vector machine and from which the optimal
            value of gamma will be chosen.
            If None, np.logspace(-3, 3, 10) is used. This range should be increased if
            the optimal gamma is one of the boundary values.
            See https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    verbose: boolean (default: False)
            If True, display the results of internal steps, such as the optimal
            parameters for the classifier.
    Returns
    -------
    res: array
            Summary statistics from the bootstrapped datasets -- the median mutual
            information and the 10% and 90% confidence limits.
            If overtime is True, each row corresponds to a different duration of the time
            series with the shortest duration, just the first time point, in the first row
            and the longest duration, the entire time series, in the last row.
    Example
    -------
    >>> import numpy as np
    >>> import MIdecoding as mi
    >>> import matplotlib.pylab as plt
    >>> res= mi.estimateMI([data1, data2], verbose= True, overtime= True)
    >>> plt.figure()
    >>> mi.plotMIovertime(res, color= "r")
    >>> plt.xlabel("time")
    >>> plt.ylabel("MI")
    >>> plt.legend()
    >>> plt.show()
    """
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = flatten(data[i][j])

    # default values
    if not Crange:
        Crange = np.logspace(-3, 3, 12)
    if not gammarange:
        gammarange = np.logspace(-3, 3, 12)

    # data is a list with one array of time series for different class
    Xos = [0]*len(data)
    ys = [0]*len(data)
    n_classes = len(data[0])
    for i in range(len(data)):
        Xos[i] = np.vstack([timeseries for timeseries in data[i]])
        ys[i] = np.hstack(
        [j * np.ones(timeseries.shape[0]) for j, timeseries in enumerate(data[i])]
            )

    if overtime:
        # loop over time series iteratively
        durations = np.arange(1, Xos[0].shape[1] + 1)
    else:
        # full time series only
        durations = [Xos[0].shape[1]]
    # array for results
    res = np.zeros((len(durations), 3))

    for j, duration in enumerate(durations):
        if j == len(durations) - 1:
            grid_pipelines = [0]*len(Xos)
            for k in range(len(Xos)):
                # slice of of time series
                if verbose:
                    print("duration of time series is", duration)
                X = Xos[k][:, :duration]
        
                # initialise scikit-learn routines
                nPCArange = range(1, X.shape[1] + 1) if X.shape[1] < 5 else [3, 4, 5]
                params = [
                    {"project__n_components": nPCArange},
                    {
                        "classifier__kernel": ["rbf"],
                        "classifier__C": Crange,
                        "classifier__gamma": gammarange,
                    },
                ]
                pipe = Pipeline(
                    [
                        ("project", PCA()),
                        ("rescale", StandardScaler()),
                        ("classifier", svm.SVC(probability=True)),
                    ]
                )
        
                # find best params for pipeline
                grid_pipelines[k] = GridSearchCV(pipe, params, n_jobs=-1, cv=5)
                grid_pipelines[k].fit(X, ys[k])
                if verbose:
                    print(grid_pipelines[k].best_estimator_)
                pipe.set_params(**grid_pipelines[k].best_params_)
        
            # find mutual information for each bootstrapped dataset
            mi = np.empty(n_bootstraps)
            for i in range(n_bootstraps):
                p = np.random.RandomState().permutation(len(ys[0]))
                test_size = np.round(0.3*len(ys[0])).astype("int")
                y_pred = np.zeros((test_size,n_classes))
                for k in range(len(ys)):
                    X = Xos[k]
                    y = ys[k]
                    X = X[p]
                    y = y[p]
                    test_size = np.round(0.3*len(y)).astype("int")
                    X_train, y_train = X[test_size:], y[test_size:]
                    X_test, y_test = X[:test_size], y[:test_size]
                    #print("Xshape",X.shape)
                    #print("Xtrain shape",X_train.shape)
                    
                    '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)'''
                    # run classifier use optimal params
                    pipe.fit(X_train, y_train)
                    #y_predict = pipe.predict(X_test)
                    y_preds = pipe.predict_proba(X_test)
                    y_pred += y_preds

                y_predict = np.argmax(y_pred,axis=1)
                # estimate mutual information
                p_y = 1 / n_classes
                p_yhat_given_y = confusion_matrix(y_test, y_predict, normalize="true")
                
                p_yhat = np.sum(p_y * p_yhat_given_y, 0)
                h_yhat = -np.sum(p_yhat[p_yhat > 0] * np.log2(p_yhat[p_yhat > 0]))
                log2_p_yhat_given_y = np.ma.log2(p_yhat_given_y).filled(0)
                h_yhat_given_y = -np.sum(
                    p_y * np.sum(p_yhat_given_y * log2_p_yhat_given_y, 1)
                )
                mi[i] = h_yhat - h_yhat_given_y
    
            # summary statistics - median and confidence intervals
            res[j, :] = [
                np.mean(np.sort(mi)[int(np.min(ci) * n_bootstraps):int(np.max(ci) * n_bootstraps)]),
                np.sort(mi)[int(np.min(ci) * n_bootstraps)],
                np.sort(mi)[int(np.max(ci) * n_bootstraps)],
            ]

            if verbose:
                print("median MI= {res[j,0]:.2f} [{res[j,1]:.2f}, {res[j,2]:.2f}]")
        else:
            pass
    return res[-1]


def plotMIovertimeS(res, color="b", label=None):
    """
    Plots the median mutual information against the duration of the time series.
    The region between the lower and upper confidence limits is shaded.
    Parameters
    ----------
    res: array
            An array of the median mutual information and the lower and upper confidence
            limits with the first row corresponding to the time series of shortest
            duration and the last two corresponding to the time series of longest
            duration.
    color: string, optional
            A Matplotlib color for both the median and shaded interquartile range.
    label: string, optional
            A label for the legend.
    """
    durations = np.arange(res.shape[0])
    plt.plot(durations, res[:, 0], ".-", color=color, label=label)
    plt.fill_between(durations, res[:, 1], res[:, 2], color=color, alpha=0.2)