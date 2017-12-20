import numpy as np

np.random.seed(1)

def non_linear(x, derivative=False):
  if(derivative==True):
    return (x*(1-x))
  
  return 1/(1+np.exp(-x))

class Network(object):
  def __init__(self, 
    input_size=3,
    intermediate_size=4,
    intermediate_depth=2,
    output_size=1,
    non_linear=non_linear,
  ):
    self.non_linear = non_linear
    self.synapses_connecting_layers = []
    self.synapses_connecting_layers.append(
      2*np.random.random((input_size,intermediate_size))-1 # input_size X intermediate_size matrix of random weights
    )
    for i in range(intermediate_depth-1):
      self.synapses_connecting_layers.append(
        2*np.random.random((intermediate_size,intermediate_size))-1
      )
    self.synapses_connecting_layers.append(
      2*np.random.random((intermediate_size,output_size))-1
    )
    
  def train(self, input_data, output_data, iterations=100000, verbose=False):
    for j in range(iterations):
      #run training input data through net and get neurons
      layers_of_neurons = self.analyze(input_data)
      
      #back propagation:

      #calculate amounts to adjust synaptic weights working backwards from associated output_data
      layer_of_neurons_error = output_data - layers_of_neurons[-1] #last layer of neurons
      if(j % 10000) == 0 and verbose:
        print("Error: {}".format(np.mean(np.abs(layer_of_neurons_error))))
      layers_of_neurons_delta = [
        layer_of_neurons_error*self.non_linear(layers_of_neurons[-1], derivative=True)
      ]

      for idx in range(len(layers_of_neurons)-2, 0, -1): #skip last layer of neurons
        layer_of_neurons_error = layers_of_neurons_delta[0].dot(
          self.synapses_connecting_layers[idx].T
        )
        layers_of_neurons_delta.insert(0,
          layer_of_neurons_error * self.non_linear(layers_of_neurons[idx], derivative=True)
        )

      for idx in range(len(self.synapses_connecting_layers)-1, -1, -1):
#idx, synapses_connecting_layer in enumerate(reversed(self.synapses_connecting_layers)):
        self.synapses_connecting_layers[idx] += layers_of_neurons[idx].T.dot(layers_of_neurons_delta[idx])
    return layers_of_neurons

  def analyze(self, input_data):
    #calculate neuron values based on prior neuron values times synaptic weights.
    layers_of_neurons = []
    layers_of_neurons.append(input_data)
    for synapses_between_two_layers in self.synapses_connecting_layers:
      layers_of_neurons.append(self.non_linear(
        np.dot(layers_of_neurons[-1], synapses_between_two_layers)
      ))
    return layers_of_neurons

def main(
  verbose,
  intermediate_size,
  intermediate_depth,
  training_data_in,
  training_data_out,
  analyze_data_in
):
  input_size = training_data_in.shape[1]
  output_size = training_data_out.shape[1]

  network = Network(
    input_size=input_size,
    intermediate_size=intermediate_size,
    intermediate_depth=intermediate_depth,
    output_size=output_size
  )
  layers_of_neurons = network.train(training_data_in, training_data_out, verbose=verbose)
  print("Result -> {}".format(str(
    network.analyze(analyze_data_in)[-1]
  )))

import sys, getopt
import pandas as pd

if __name__ == "__main__":
  help = (
    'usage: {} [options] [-s size |-d depth] -i index filename_training_csv analyze_data_in\n\n'
    'where:\n'
    '    size = size of intermediate node layers. Defaults to 3 if not provided.\n'
    '    depth = depth of intermediate node layers. Defaults to 1 if not provided.\n'
    '    index = column index (zero based) in filename_training_csv from which to slice training results\n'
    '    filename_training_csv = name of file with comma-delimited training data where [:index] is training input and [index:] is training output. First line is ignored.\n'
    '    analyze_data_in = comma delimited list of ints, e.g. 1,0,0\n\n'
    'Options\n'
    '    -v     verbose output\n'.format(sys.argv[0])
  )
  size = 3
  depth = 1
  verbose = False
  index = None
  try:
    (opts, args) = getopt.getopt(sys.argv[1:], "hvs:d:i:")
  except getopt.GetoptError:
    print(help)
    sys.exit(2)
  if len(args) < 2:
    print(help)
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print(help)
      sys.exit()
    elif opt in ('-s',):
      size = int(arg)
    elif opt in ('-d',):
      depth = int(arg)
    elif opt in ('-v',):
      verbose = True
    elif opt in ('-i',):
      index = int(arg)
  if index is None:
    print(help)
    sys.exit(2)
  #read data
  df_training = pd.read_csv(args[0], header=None, skiprows=[0])
  training_data_in = df_training.iloc[:,:index].values
  training_data_out = df_training.iloc[:,index:].values

  analyze_data_in = np.array(list(map(int, args[1].split(','))))

  main(
    verbose,
    size,
    depth,
    training_data_in,
    training_data_out,
    analyze_data_in
  )
