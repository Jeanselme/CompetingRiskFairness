# This script extracts the different fold to then use the fine gray R script
import sys
from generate import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split

# Select random seed
random_seed = int(sys.argv[1])

print("Script running experiments on generated data with seed =", random_seed)
x, t, e, _, _ = generate(random_seed) 

# Normalise data
x, t, e = StandardScaler().fit_transform(x.values).astype(float),\
            t.values.astype(float),\
            e.values.astype(int)

## Save data for R 
kf = ShuffleSplit(n_splits = 1, random_state = random_seed, test_size = 0.2)
data = pd.DataFrame(x).add_prefix('feature') # Do not save names to match R

for i, (train_index, test_index) in enumerate(kf.split(x, e)):
    train_index, dev_index = train_test_split(train_index, test_size = 0.2, random_state = random_seed, stratify = e[train_index])
    dev_index, val_index   = train_test_split(dev_index,   test_size = 0.5, random_state = random_seed, stratify = e[dev_index])

    # Keep track of the whole indexing
    fold = pd.Series(0, index = data.index)
    fold[train_index] = "Train"
    fold[dev_index] = "Dev"
    fold[val_index] = "Val"
    fold[test_index] = "Test"
    data['Fold_{}'.format(i)] = fold

data['Time'] = t
data['Event'] = e
data.to_csv('tmp.csv', index = False)