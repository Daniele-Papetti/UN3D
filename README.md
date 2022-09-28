# UN<sup>3</sup>D

## How to install UN<sup>3</sup>D
1. Install its dependencies
	- numpy, matplotlib, pandas, scikit-learn;
	- keras;
	- minisom, refer to the [original repository](https://github.com/JustGlowing/minisom); 
	- SMAC3, refer to the installation section of the [original repository](https://github.com/automl/SMAC3).
2. Clone the repository

## Hot to use UN<sup>3</sup>D
```
from som import train_and_evaluate
from autoencoder import signals_embedding
from hpo import optimization, best_config


if __name__ == "__main__":
	# load the signals, which is a list containing the signals to cluster
	signals_embedding(signals, "embeddings.pkl")
	n_cluster = 10
	optimization("embeddings.pkl", n_cluster, "labels.pkl", 60, 40, 60, 50)
	best_config("best_seens.pkl")
	# the best configuration file can be manually updated or 
	# a new file can be generated starting from the returned
	# configuration of best_config
	train_and_evaluate("examples/hp_conf.txt", "embeddings.pkl", True,
	                   n_cluster, "labels.pkl")
```