import glob
import numpy as np
from keras.layers import Dense, Convolution1D, MaxPooling1D, Flatten
from keras.models import Sequential
from sklearn.cross_validation import StratifiedKFold

# Number of folds to use for k-fold cross validation
K_FOLDS = 10
# Number of features in the word feature vectors from GloVe (should be 50, 100, 200, or 300)
WORD_FEATURES = 100
# Number of words per document to normalize all documents to
DOC_LENGTH = 1024

# Initialize dictionary for holding word vectors with words as keys, and their corresponding vectors as values
wordVectors = {}

print("Loading word vectors...")

# Open appropriate GloVe file
f = open("./glove.6B/glove.6B.XXd.txt".replace("XX", str(WORD_FEATURES)))
# Iterate through all lines (words) in the file
for line in f:
    # Split each line
    values = line.split()
    # If nothing results from the split, skip this line
    if len(values) == 0:
        continue

    # Take the first element as the word (to use as the key)
    word = values[0]
    # Take the remaining elements as a vector of floats (to use as the value)
    coefs = np.asarray(values[1:], dtype='float32')
    # Add key and value to dictionary
    wordVectors[word] = coefs
# Close the GloVe file
f.close()

# Initialize arrays for holding the raw document data as well as their labels
rawDocuments = []
labels = []

print("Loading files...")

# Iterate through all files in the input folder
for filename in glob.glob("./review_polarity/txt_sentoken/*/*.txt"):
    # Open the file
    f = open(filename)

    # Load the text content as a whole string into the list of raw documents
    rawDocuments.append(f.read())
    # Add the label of this text to the list of labels
    labels.append(0 if 'neg' in filename else 1)

    # Close the file
    f.close()

print("Processing files...")

# Initialize 3-dimensional numpy array for holding data to feed to the network, initializing all values as zeroes
# First dimension = The documents which serve as the individual labelled elements
# Second dimension = The words that make up each document
# Third dimension = The elements of the vectors that represent each word
vectorDocs = np.zeros((len(rawDocuments), DOC_LENGTH, WORD_FEATURES))

# Iterate through all raw documents
for currDoc in range(len(rawDocuments)):
    # Replace sentence-marking newlines with spaces
    document = rawDocuments[currDoc].replace("\n", " ")

    # Split the data into a list of tokens
    all_tokens = document.split(" ")
    # Remove empty tokens
    tokens = [token for token in all_tokens if token != ""]

    # If there are more tokens than the document size threshold, crop the list of tokens
    if len(tokens) > DOC_LENGTH:
        tokens = tokens[:DOC_LENGTH]

    # Iterate through all tokens
    for currToken in range(len(tokens)):
        token = tokens[currToken]

        # Get the word vector that represents this token, if it exists
        wordVector = wordVectors.get(token)

        # If it does exist, replace the vector at this word's position
        if wordVector is not None:
            vectorDocs[currDoc][currToken] = wordVector
        # Otherwise, if it doesn't exist, leave it as a vector of 0s

print("Building network...")


def build_model(input_shape, layer_sizes, filter_sizes, final_pool_size, dense_size):
    """
    Builds a convolutional network model with the given parameters, and compiles it.

    :param input_shape: Shape of the input vector
    :param layer_sizes: List of output sizes of the convolutional layers.  If one layer is smaller than the previous,
    a max-pooling layer is added in between to compensate
    :param filter_sizes: List of filter lengths of the convolutional layers
    :param final_pool_size: Output length after the final pooling layer
    :param dense_size: Number of outputs in penultimate densely-connected layer
    :return: The constructed, compiled model
    """
    # Get the output size of the last convolutional layer
    last_output_size = layer_sizes[len(layer_sizes)-1]

    # If the input size to the first dense layer is greater than the output size of the final convolutional/pooling
    # layer, adjust the input size appropriately
    if final_pool_size > last_output_size:
        final_pool_size = last_output_size

    # Create the sequential model
    m = Sequential()

    # Iterate through all layers
    for layer in range(len(layer_sizes)):
        # If this is the first layer, add the initial convolutional layer with defined input shape
        if layer == 0:
            m.add(Convolution1D(layer_sizes[0], filter_sizes[0], activation='relu', input_shape=input_shape))
        # Otherwise, add a pooling layer (if output size of this layer is less than the previous), and add the
        # next convolutional layer
        else:
            if layer_sizes[layer] < layer_sizes[layer-1]:
                m.add(MaxPooling1D(int(layer_sizes[layer-1] / layer_sizes[layer])))
            m.add(Convolution1D(layer_sizes[layer], filter_sizes[layer], activation='relu'))

    # If the output size of the last convolutional layer is greater than the input size of the first dense layer,
    # pool the outputs appropriately
    if last_output_size > final_pool_size:
        m.add(MaxPooling1D(int(last_output_size / final_pool_size)))

    # Flatten the multi-dimensional output of the convolutional layers to a single-dimensional vector to feed to the
    # dense layers
    m.add(Flatten())

    # Add the first dense layer, with the defined output size
    m.add(Dense(dense_size, activation='relu'))

    # Add the final dense layer with a binary output, with sigmoid activation
    m.add(Dense(1, activation='sigmoid'))

    # Compile the model using 'rmsprop' as an optimization method
    m.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Return the constructed model
    return m


def train_and_test(m, x, y, folds):
    """
    Tests the given model with the given input and label data, using the given k-fold cross validation

    :param m: The model to use
    :param x: The input data
    :param y: The label data
    :param folds: The k-fold cross-validation information
    :return: The list of accuracy results from the cross-validation
    """
    # Initialize list for holding accuracy results from each k-fold
    fold_acc = []

    # Iterate through all k-folds
    for trn_i, test_i in folds:
        print("Fold #" + str(len(fold_acc) + 1) + ":")
        print("------------------")
        # Train the model
        m.fit(x[trn_i], y[trn_i], nb_epoch=2)
        # Test the model and get resulting accuracy
        _, acc = m.evaluate(x[test_i], y[test_i])

        print("Fold accuracy = " + str(acc))
        print()

        # Add accuracy result to list of results
        fold_acc.append(acc)

    return fold_acc

#

# DEFINING MODEL PARAMETERS TO TRY

#

# Create list of model parameters to try (explained below)
modelParams = [
    {
        "layerSizes": [256, 256],
        "filterSizes": [4, 4],
        "finalPool": 2,
        "denseSize": 128
    }
    ]
'''
    ,  Other good models
    {
        "layerSizes": [256, 256],
        "filterSizes": [4, 4],
        "finalPool": 2,
        "denseSize": 512
    },
    {
        "layerSizes": [256, 64],
        "filterSizes": [4, 8],
        "finalPool": 2,
        "denseSize": 512
    },
    {
        "layerSizes": [128, 64],
        "filterSizes": [8, 8],
        "finalPool": 4,
        "denseSize": 512
    },
    {
        "layerSizes": [64, 32],
        "filterSizes": [2, 2],
        "finalPool": 2,
        "denseSize": 1024
    },
    {
        "layerSizes": [128, 64],
        "filterSizes": [4, 8],
        "finalPool": 8,
        "denseSize": 1024
    }
]
'''

'''
modelParams = {
    layerSizes = The list of the output sizes of the convolutional layers.  The length of this list is the # of layers
    filterSizes = The list of filter lengths of the convolutional layers.  The length of this list is the # of layers
    finalPool = The length of the vectors before passing to dense layers (the size after the final pooling)
    denseSize = The output size of the first dense layer.
}

For example,
modelParams = {
    layerSizes = [128, 64, 32],
    filterSizes = [4, 8, 16],
    finalPool = 8,
    denseSize = 512
}
results in a model defined as:

model = Sequential()
model.add(Convolution1D(128, 4, activation='relu', input_shape=(DOC_LENGTH, WORD_FEATURES)))
model.add(MaxPooling1D(2))
model.add(Convolution1D(64, 8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Convolution1D(32, 16, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
'''


print("Fitting and evaluating network...")
print()

# Create the k-fold cross validation info
kFold = StratifiedKFold(labels, K_FOLDS)

# Convert the labels to a numpy array
labels = np.array(labels)

# Define the input shape of the first convolutional layer
inputShape = (DOC_LENGTH, WORD_FEATURES)

# Initialize model variable
model = None

# Initialize max-trackers and counters
modelCount = 0

# Iterate through all model parameters
for modelParam in modelParams:
    # Delete the old model (probably not necessary)
    del model

    # Increment model counter
    modelCount += 1
    print()
    print("NEW MODEL (#" + str(modelCount) + ") = " + str(modelParam))

    # Build the model using the parameters stored in this modelParam object
    model = build_model(inputShape,
                        modelParam.get("layerSizes"),
                        modelParam.get("filterSizes"),
                        modelParam.get("finalPool"),
                        modelParam.get("denseSize"))

    # Train and test the model with the dataset, collecting the accuracy results and converting to numpy array
    accResults = train_and_test(model, vectorDocs, labels, kFold)
    accResults = np.array(accResults)

    # Get the mean accuracy as well as the best fold accuracy
    accMean = accResults.mean()
    accBest = accResults.max()

    print("AVERAGE ACCURACY = " + str(accMean))




















