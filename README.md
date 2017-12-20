# neural net experiments

A simple neural net with data to demonstrate learning and prediction.

### Algorithm

* a sigmoid function as its non-linearity
* a simple back propagation mechanism to train synaptic weights based on the error weighted by the derivative of the non-linearity. Errors for small derivatives make little changes to synaptic weights whereas for large derivatives make larger changes to synaptic weights.

### Dependencies

*  numpy
*  pandas

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

### Results

*  training_index-3_110-1.txt: Training description: output opposite of third input. Analyzing [1, 1, 0]: Expect 1. Converges and successfully predicts with net of intermediate node size of 3 and depth of 1, and intermediate node size of 6 and depth of 3.
*  training_index-3_111-0.txt: Training description: output 1 if first and second inputs are opposite. Analyzing [1, 1, 1]: Expect 0.  Converges and successfully predicts with net of intermediate node size of 3 and depth of 1, and intermediate node size of 20 and depth of 1.
*  training_index-3_100-10.txt: Training escription: opposite in position 1 and 2 outputs 1 and 1 in position 3 outputs second 1. Analyzing [1,0,0]: expect [1,0]. Converges and successfully predicts with net of intermediate node size of 3 and depth of 1, int_size of 20 and depth of 1.
* training_index-3_100-01.txt: Training description: opposite in position 1 and 2 outputs 1 and 1 in position 3 outputs second 1. Analyzing [1, 0, 0]: Expect [0,1]. Converges and successfully predicts with net of intermediate node size of 3 and depth of 1, and intermediate node size of 20 and depth of 1.
*  training_index-3_100-1.txt: Training description: output 1 if first is one. Analyze [1, 0, 0]: Expect 1.  Converges and successfully predicts with net of intermediate node size of 3 and depth of 1, and intermediate node size of 20 and depth of 1.
*  training_index-4_1100-1.txt: Training description: two consecutive 1s are 1. Analyzing [1, 1, 0, 0]: Expect 1. Cannot find net that works. Successfully converged with intermediate node size and depths of: 4,1; 4,2; 6,6; 20,1 yet answer was decisively the opposite of what was predicted (0 instead of 1). Appears that net is trained on another pattern I don't see, or its structure or back propagation mechanism can't be used to train successfully on this pattern.
