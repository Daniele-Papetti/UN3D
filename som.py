import time
import pickle
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score


def get_hyperparameters(file):
    '''

    Extracts the hyperparameters configuration from a tab separeted file,
    each row must contain the hyperparameter and its value.
    The hyperparameters are the following:
        dim -> the length of the side of the SOM;
        sigma -> sigma parameter used in the neighbourhood functions;
        lr -> learning rate used to update the SOM weights;
        neigh_f -> neighbourhood function, the possbile values are 'gaussian', 'bubble', 'triangle', 'mexican_hat';
        dist_f -> activation function, possible values are 'euclidean', 'cosine', 'manhattan', 'chebyshev';
        topology -> 'rectangular' or hexagonal;

    Parameters
    ----------
    file : String
        path to the file to load the hyper-parameter configuration.

    Returns
    -------
    hp_conf : Dict
        dictionary whose keys are the SOM's hyper-parameters and the values are the
        actuale values of the respective hyper-parameter.

    '''
    df = pd.read_csv(file, sep='\t', header='infer', names=['key', 'value'])
    hparams = ['dim', 'sigma', 'lr', 'neigh_f', 'dist_f', 'topology']
    hp_conf = dict()
    for h in hparams:
        try:
            hp_conf[h] = df.loc[df['key'] == h, 'value'].item()
        except:
            print("Missing hyper-parameters! The hyper-parameters are: dim, sigma, lr, neigh_f, dist_f, topology")
            raise
    hp_conf['dim'] = int(hp_conf['dim'])
    hp_conf['sigma'] = float(hp_conf['sigma'])
    hp_conf['lr'] = float(hp_conf['lr'])
    return hp_conf


def train_som(embedding_file, hp_conf, label,
              label_file=None,
              dump_som_file="som.pkl",
              order_y_file="ordered-y.pkl",
              weights_file="weights.pkl",
              map_bmu_signal_file="ordered_unit_pred.pkl",
              labels_pred_file="labels_pred.pkl"):
    '''

    Trains the SOM using the whole dataset. Once the SOM is trained, the Best
    Matching Unit of each signal is predicted.
    You should use label and label_file when the ground truth is available and
    you want to visualize the clustering performances.

    Parameters
    ----------
    embedding_file : String
        pickle file containing the embeddings.
    hp_conf : Dict
        dictionary containing the hyperparameters, it can be created using the
        get_hyperparameters function.
    label : Bool
        if True label_file has to be provided, the predictions and labels will be
        ordered according to the ground_truth.
    label_file : String or list, optional
        a path or a list of a paths that contain the ground truth of the
        embedding file. Such files has to be stored as pickle files.
        The default is None.
    dump_som_file : String, optional
        a path where to dump the trained SOM, it is saved as a pickle file.
        The default is "som.pkl".
    order_y_file : String, optional
        a path where to dump the ground truth ordered according to
        the labels, it is saved as a pickle file. It will be used only if label is True and label_file is not None.
        The default is "ordered-y.pkl".
    weights_file : String, optional
        a path where to dump the weights of each SOM unit, it is saved as a pickle file.
        The default is "weights.pkl".
    map_bmu_signal_file : String, optional
        a path where to save a dictionary mapping each SOM unit with a list containing
        the IDs of all the embeddings represented by that unit, it is saved as a pickle file.
        The default is "ordered_unit_pred.pkl".
    labels_pred_file : String, optional
        a path where to save the predictions of each embedding caontained in embedding file.
        If label is False or label_file is None, the i-th label represent the BMU
        of the i-th embedding of the embedding file.
        If label is True and label_file is not None, the i-th label represent the BMU
        of the i-th signal after the ordering process.
        In any case the file is saved as a pickle file. The default is "labels_pred.pkl".

    Returns
    -------
    labels_pred : List
        List containing the BMUs of each embedding .

    '''
  
    with open(embedding_file, 'rb') as f:
        x = pickle.load(f)
    dim0 = x[0].shape[0]
    dim1 = x[0].shape[1]
    # flattening the embedding
    x = [[e for sl in list(map(list, zip(*s))) for e in sl] for s in x]

    som = MiniSom(hp_conf['dim'], hp_conf['dim'], input_len=dim0 * dim1,
                  sigma=hp_conf['sigma'], learning_rate=hp_conf['lr'],
                  neighborhood_function=hp_conf['neigh_f'],
                  activation_distance=hp_conf['dist_f'],
                  topology=hp_conf['topology'])
    start_time = time.time()
    som.pca_weights_init(x)
    som.train(x, num_iteration=len(x), random_order=True, verbose=False)
    neurons = som.get_weights()
    end_time = time.time()
    print(" Time: {}".format(round(end_time - start_time, 2)))

    if dump_som_file:
        with open(dump_som_file, 'wb') as f:
            pickle.dump(som, f)

    if weights_file:
        with open(weights_file, 'wb') as f:
            pickle.dump(neurons, f)

    # if label is true, the evaluation of metrics will be done
    if label and label_file:
        # load labels
        if isinstance(label_file, str):
            with open(label_file, 'rb') as f:
                y = pickle.load(f)
        elif isinstance(label_file, list):
            y = list()
            for dp in label_file:
                with open(dp, 'rb') as f:
                    y += pickle.load(f)
        # order label and signals
        ordered_x = list()
        ordered_y = list()
        map_y_xs = {k: list() for k in y}
        for i in range(len(y)):
            map_y_xs[y[i]].append(x[i])
        for k in map_y_xs:
            ordered_x += map_y_xs[k]
            ordered_y += [k] * len(map_y_xs[k])
        with open(order_y_file, 'wb') as f:
            pickle.dump(ordered_y, f)
        # predicting with ordered x
        labels_pred = [som.winner(e) for e in ordered_x]
        # mapping BMU and signals
        map_bmu_preds = {(i, j): list() for i in range(hp_conf['dim']) for j in range(hp_conf['dim'])}
        for i in range(len(labels_pred)):
            map_bmu_preds[labels_pred[i]].append(i)
        with open(map_bmu_signal_file, 'wb') as f:
            pickle.dump(map_bmu_preds, f)
    else:
        labels_pred = [som.winner(e) for e in x]

    with open(labels_pred_file, 'wb') as f:
        pickle.dump(labels_pred, f)

    return labels_pred


def cluster_som_units(n_clusters, hp_conf, labels_pred,
                      weights_file="weights.pkl"):
    '''

    clusters the SOM units and then assign to each embedding the ID of the
    cluster it belongs to. It returns a list where the i-th label is the cluster
    the i-th embedding is assigned to, the i-th element is refered to the i-th
    embedding in embedding_file passed to the somming_embedding function.

    Parameters
    ----------
    n_clusters : int
        number of clusters.
    hp_conf : Dict
        dictionary containing the hyperparameters, it can be created using the
        get_hyperparameters function.
    labels_pred : String or list
        if it's a string, it is a path where the predictions are saved.
        If it's a list, the element contained are interpreted as the predicted labels
        returned by the somming_embedding function.
    weights_file : String, optional
        a path where to load the weights of each SOM unit,
        the file has to be a pickle file.
        The default is "weights.pkl".

    Returns
    -------
    signal_clusters : list
        list witht he i-th element containing the cluster assigned to the i-th embedding.

    '''
    BMUs = [(i, j) for i in range(hp_conf['dim']) for j in range(hp_conf['dim'])]
    clf = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    with open(weights_file, 'rb') as f:
        neurons = pickle.load(f)
    weights = [neurons[coord] for coord in BMUs]
    clusters = clf.fit_predict(weights)
    # assign to each bmu its cluster
    map_bmu_cluster = dict()
    for bmu, cluster in zip(BMUs, clusters):
        map_bmu_cluster[bmu] = cluster
    # mapping signal to cluster
    signal_clusters = list()
  
    if isinstance(labels_pred, str):
        with open(labels_pred, 'rb') as f:
            labels_pred = pickle.load(f)
    for s_bmu in labels_pred:
        signal_clusters.append(map_bmu_cluster[s_bmu])
  
    return signal_clusters


def evaluate_performance(n_clusters, hp_conf, labels_pred, ground_truth_file,
                         weights_file="weights.pkl"):
    '''

    Clusters the embeddings using the SOM units, then it eveluates the performances
    in terms of Adjusted Random Score (ars) and Fowlkes-Mallows Score (fms).

    Parameters
    ----------
    n_clusters : int
        number of clusters.
    hp_conf : Dict
        dictionary containing the hyperparameters, it can be created using the
        get_hyperparameters function.
    labels_pred : String or list
        if it's a string, it is a path where the predictions are saved.
        If it's a list, the element contained are interpreted as the predicted labels
        returned by the somming_embedding function.
    ground_truth_file : String
        a path to a pickle file containing a list where the i-th element is the
        ground truth cluster of the i-th signal.
    weights_file : String, optional
        a path where to load the weights of each SOM unit,
        the file has to be a pickle file.
        The default is "weights.pkl".

    Returns
    -------
    ars : float
        Adjusted Random Score.
    fms : float
        Fowlkes-Mallows Score.

    '''

    signal_clusters = cluster_som_units(n_clusters, hp_conf, labels_pred,
                                        weights_file=weights_file)
    with open(ground_truth_file, 'rb') as f:
        ground_truth = pickle.load(f)
    fms = fowlkes_mallows_score(ground_truth, signal_clusters)
    print(" Fowlkes-Mallows score: {}".format(fms))
    ars = adjusted_rand_score(ground_truth, signal_clusters)
    print(" Adjusted Rand score: {}".format(ars))
  
    return ars, fms


def train_and_evaluate(hyperparameter_file, embedding_file, label,
                       n_clusters, label_file=None,
                       dump_som_file="som.pkl",
                       order_y_file="ordered-y.pkl",
                       weights_file="weights.pkl",
                       map_bmu_signal_file="ordered_unit_pred.pkl",
                       labels_pred_file="labels_pred.pkl"):
    '''

    Wrapper to train the SOM, predict the cluster of each embedding and evaluate
    the performances in terms of Adjusted Random Score (ars)
    and Fowlkes-Mallows Score (fms).

    Parameters
    ----------
    file : String
        path to the file to load the hyper-parameter configuration.
    embedding_file : String
        pickle file containing the embeddings.
    label : Bool
        if True label_file has to be provided, the predictions and labels will be
        ordered according to the ground_truth.
    n_clusters : int
        number of clusters.
    label_file : String or list, optional
        a path or a list of a paths that contain the ground truth of the
        embedding file. Such files has to be stored as pickle files.
        The default is None.
    dump_som_file : String, optional
        a path where to dump the trained SOM, it is saved as a pickle file.
        The default is "som.pkl".
    order_y_file : String, optional
        a path where to dump the ground truth ordered according to
        the labels, it is saved as a pickle file. It will be used only if label is True and label_file is not None.
        The default is "ordered-y.pkl".
    weights_file : String, optional
        a path where to dump the weights of each SOM unit, it is saved as a pickle file.
        The default is "weights.pkl".
    map_bmu_signal_file : String, optional
        a path where to save a dictionary mapping each SOM unit with a list containing
        the IDs of all the embeddings represented by that unit, it is saved as a pickle file.
        The default is "ordered_unit_pred.pkl".
    labels_pred_file : String, optional
        a path where to save the predictions of each embedding caontained in embedding file.
        If label is False or label_file is None, the i-th label represent the BMU
        of the i-th embedding of the embedding file.
        If label is True and label_file is not None, the i-th label represent the BMU
        of the i-th signal after the ordering process.
        In any case the file is saved as a pickle file. The default is "labels_pred.pkl".

    Returns
    -------
    labels_pred: list
        List containing the corresponding cluster for each signal.
    ars : float
        Adjusted Random Score.
    fms : float
        Fowlkes-Mallows Score.

    '''
    print(" *** Loading Hyper-parameters ***")
    hp_conf = get_hyperparameters(hyperparameter_file)
    print(" *** Training the SOM ***")
    labels_pred = train_som(embedding_file, hp_conf, label, label_file=label_file, dump_som_file=dump_som_file,
                            order_y_file=order_y_file, weights_file=weights_file,
                            map_bmu_signal_file=map_bmu_signal_file, labels_pred_file=labels_pred_file)
    if label_file is not None:
        print(" *** Evaluating performance ***")
        ars, fms = evaluate_performance(n_clusters, hp_conf, labels_pred,
                                        label_file, weights_file=weights_file)

        return labels_pred, ars, fms
    else:
        return labels_pred
