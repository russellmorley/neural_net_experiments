# neural net experiments

A simple neural net with data to demonstrate learning.

### Algorithm

* a sigmoid function as its non-linearity
* a simple back propagation mechanism to train synaptic weights based on the error weighted by the derivative of the non-linearity. Errors for small derivatives make little changes to synaptic weights whereas for large derivatives make larger changes to synaptic weights.

### Usage

```
usage: nn.py [options] [-s size |-d depth] -i index filename_training_csv analyze_data_in

where:
  size = size of intermediate node layers. Defaults to 3 if not provided.
  depth = depth of intermediate node layers. Defaults to 1 if not provided.
  index = column index (zero based) in filename_training_csv from which to slice training results
  filename_training_csv = name of file with comma-delimited training data where [:index] is training input and [index:] is training output. First line is ignored
  analyze_data_in = comma delimited list of ints, e.g. 1,0,0

Options
  -v     verbose output
```

#### Example command

To configure the neural net to have intermediate layers of size 3 and depth of 1, train on training_index-3_110-1.txt, then analyze 1,1,0 to predict the result:

    python nn.py -s 3 -d 1 -i 3 -v training_index-3_110-1.txt 1,1,0

