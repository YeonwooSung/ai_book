# Surprise

The surprise is a Python scikit for building and analyzing recommender systems that deal with explicit rating data. It is designed to be easy to use, fast, and extensible.

## Installation

```bash
pip install surprise
```

## Usage

```python
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

For more usage, please refer to either [this page](https://surprise.readthedocs.io/en/stable/getting_started.html) or [this notebook](./surprise_tutorial.ipynb)
