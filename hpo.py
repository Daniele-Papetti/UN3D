import os
import sys
import time
import shutil
import pickle
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib.gridspec import GridSpec
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace, Configuration, Constant
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace import InCondition, ForbiddenEqualsClause, ForbiddenAndConjunction
# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer.acquisition import LCB
from smac.initial_design import latin_hypercube_design
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.runhistory.runhistory import RunHistory


def fitness(config):
    '''
  
    Evaluates the fitness of the hyperparameter configuration stored in config.
    This function is supposed to be called only by the optimization process.
  
    Parameters
    ----------
    config : Dictionary
        a dictionary where the keys are the hyperparameters and the values are the
        respective values.

    Returns
    -------
    fitness: float
        the evaluated value fitness of the config

    '''
    SOM_DIM = config['som_dim']
    LR = config['lr']
    NR_FN = 'gaussian'
    ACT_DIST = config['activation_distance']
    TOPOLOGY = config['topology']
    SIGMA = config['sigma']

    # input
    with open(embedding_file, 'rb') as f:
        x = pickle.load(f)
    dim0 = x[0].shape[0]
    dim1 = x[0].shape[1]
    x = [[e for sl in list(map(list, zip(*s))) for e in sl] for s in x]

    # SOMming
    som = MiniSom(SOM_DIM, SOM_DIM, input_len=dim0 * dim1, sigma=SIGMA,
                  learning_rate=LR, neighborhood_function=NR_FN,
                  activation_distance=ACT_DIST, topology=TOPOLOGY)
    som.pca_weights_init(x)
    som.train(x, num_iteration=len(x), random_order=True, verbose=False)
    neurons = som.get_weights()

    # predicting
    labels_pred = [som.winner(e) for e in x]

    #clustering
    BMUs = [(i, j) for i in range(SOM_DIM) for j in range(SOM_DIM)]
    clf = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    weights = [neurons[coord] for coord in BMUs]
    clusters = clf.fit_predict(weights)

    map_bmu_cluster = dict()
    for bmu, cluster in zip(BMUs, clusters):
        map_bmu_cluster[bmu] = cluster

    signal_clusters = list()
    for s_bmu in labels_pred:
        signal_clusters.append(map_bmu_cluster[s_bmu])

    print("evaluating performances")
    # load labels
    if isinstance(label_file, str):
        with open(label_file, 'rb') as f:
            y = pickle.load(f)
    elif isinstance(label_file, list):
        y = list()
        for dp in label_file:
            with open(dp, 'rb') as f:
                y += pickle.load(f)
      
    fms = fowlkes_mallows_score(y, signal_clusters)
    ars = adjusted_rand_score(y, signal_clusters)

    return 1 - (0.5 * fms + 0.5 * ars)


def optimization(embed_file, n_cluster, ground_truth_file, budget,
                 dim_min, dim_max, dim_def,
                 lr_min=0.3, lr_max=5.0, lr_def=0.5,
                 sigma_min=0.5, sigma_max=3.0, sigma_def=1):
    '''

    Performs the optimization process.

    Parameters
    ----------
    embed_file : String
        pickle file containing the embeddings.
    n_cluster : int
        number of clusters.
    ground_truth_file : String
        path to the pickle file containing the ground truth.
        The i-th element of the list stored in the file is the cluster
        the i-th signal belongs to.
    budget : int
        number of candidate configurations to test.
    dim_min : int
        minimum dimension of the side of the SOM.
    dim_max : int
        maximum dimension of the side of the SOM.
    dim_def : int
        default dimension of the side of the SOM.
    lr_min : float
        minimum value of the learning rate.
    lr_max : float
        maximum value of the learning rate.
    lr_def : float
        default value of the learning rate.
    sigma_min : float
        minimum value of the sigma.
    sigma_max : float
        maximum value of the sigma.
    sigma_def : float
        default value of the sigma.

    '''
    # Creating configuration space
    cs = ConfigurationSpace()
    som_dim = CategoricalHyperparameter('som_dim', [i for i in range(dim_min, dim_max)],
                                        default_value=dim_def)
    lr = UniformFloatHyperparameter('lr', lr_min, lr_max, default_value=lr_def)
  
    # warnings occured and the quantization error was nan
    sigma = UniformFloatHyperparameter('sigma', sigma_min, sigma_max, default_value=sigma_def)
  
    act_dst = CategoricalHyperparameter('activation_distance', ['euclidean', 'cosine', 'manhattan', 'chebyshev'],
                                        default_value='euclidean')
    topology = CategoricalHyperparameter('topology', ['rectangular', 'hexagonal'],
                                         default_value='rectangular')
  
    cs.add_hyperparameters([som_dim, sigma, lr, act_dst, topology])
  
    global embedding_file, n_clusters, label_file
    embedding_file = embed_file
    n_clusters = n_cluster
    label_file = ground_truth_file
    # define scenario
    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": budget,
                         "cs": cs,
                         "deterministic": "True"
                         })
  
    s_smac = SMAC4HPO(scenario=scenario, tae_runner=fitness,
                      acquisition_function=LCB,
                      initial_design=latin_hypercube_design.LHDesign,
                      initial_design_kwargs={'max_config_fracs': 1 / 6,
                                             'n_configs_x_params': 2})
    s_incumbent = s_smac.optimize()
  
    # dump results
    cfg_score = list()
    bs = list()
    best_seen = 1 - s_smac.get_runhistory().get_cost(s_smac.get_runhistory().get_all_configs()[0])
    for config in s_smac.get_runhistory().get_all_configs():
        loss = s_smac.get_runhistory().get_cost(config)
        score = 1 - loss
        cfg_score.append((config, score))
        if score > best_seen:
            best_seen = score
        bs.append(best_seen)
  
    f = open("cfgs_score.pkl", "wb")
    pickle.dump(cfg_score, f)
    f.close()
    f = open("best_seens.pkl", "wb")
    pickle.dump(bs, f)
    f.close()


def plot_best_seen(best_seen_file, color='blue'):
    '''

    Plots the best seen value until the i-th iteration of the optimization process.

    Parameters
    ----------
    best_seen_file : String
        path to the best seen file created by the optimization process.
    color : String
        a matplotlib color to draw the line. Deafault value is blue.

    '''
    with open(best_seen_file, 'rb') as f:
        bs = pickle.load(f)
    x = list(range(len(bs)))
    plt.figure(figsize=(8, 6))
    plt.plot(x, bs, '-', color=color, linewidth=1.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Best seen", fontsize=18)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.savefig("best-seen.pdf")
    plt.savefig("best-seen.png", dpi=300)
    plt.close()


def plot_cfg(cfg_file, color='blue'):
    '''

    Plots the fitness value of the i-th iteration of the optimization process.

    Parameters
    ----------
    cfg_file : String
        path to the configuration file created by the optimization process.
    color : String
        a matplotlib color to draw the line. Deafault value is blue.

    '''
    with open(cfg_file, 'rb') as f:
        cfg_score = pickle.load(f)
    x = list(range(len(cfg_score)))
    plt.figure(figsize=(8, 6))
    Y = [e[1] if e[1] > 0 else 0 for e in cfg_score]
    plt.plot(x, Y, color=color, linewidth=1.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Score of configuration evaluation", fontsize=18)
    plt.xlabel("Configuration", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.savefig("score-iteration.pdf")
    plt.savefig("score-iteration.png", dpi=300)
    plt.close()


def best_config(cfg_file):
    '''

    Prints the best configuration found the optimization process.

    Parameters
    ----------
    cfg_file : String
        path to the best seen file created by the optimization process.

    '''
    with open(cfg_file, 'rb') as f:
        cfg_score = pickle.load(f)
    tmp = max(cfg_score, key=itemgetter(1))
    print("Best configuration was \n{}\n with a score of {}\n\n".format(tmp[0], tmp[1]))
