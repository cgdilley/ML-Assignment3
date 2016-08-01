import glob
import numpy as np
from keras.layers import Dense, Convolution1D, MaxPooling1D, Flatten
from keras.models import Sequential
from sklearn.cross_validation import StratifiedKFold

# Number of folds to use for k-fold cross validation
K_FOLDS = 5
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

print("Generating model parameters...")

modelParams = []

# Permutate through a bunch of model parameters to build unique combinations to test

# Number of convolutional layers, 2-4
for layer_depth in range(2, 5):
    # Number of outputs from first dense layer, 2^(7-10)
    for dense_size_exp in range(7, 11):
        # Size of output from last convolutional/pooling layer, as input into first dense layer, 2^(1-3)
        for final_pool_size_exp in range(1, 4):
            # Size of the output from the first convolutional layer, 2^(6-8)
            for initial_layer_size_exp in range(6, 9):
                # Size of the filter length of the first convolutional layer, 2^(1-3)
                for initial_filter_size_exp in range(1, 4):
                    # Multiplier for filter lengths in successive convolutional layers, 2^(0-1)
                    for filter_size_mult_exp in range(0, 2):

                        # Create placeholder list of layer sizes all equal to the initial layer output
                        layerSizes = [2 ** initial_layer_size_exp] * layer_depth
                        # Create list for filter sizes with just the initial filter size
                        filterSizes = [2 ** initial_filter_size_exp]

                        # Iterate through the number of non-initial layers, and add the filter size for each layer
                        # based on the filter multiplier
                        for depth in range(1, layer_depth):
                            filterSizes.append(filterSizes[depth - 1] * (2 ** filter_size_mult_exp))

                        # Create some number of random configurations of the remaining layers
                        for i in range(int((initial_layer_size_exp-4) * layer_depth)):
                            # Copy layer size list into a new list to modify
                            layerSizes_ = layerSizes[:]
                            # Iterate through the number of non-initial layers
                            for depth in range(1, layer_depth):
                                # Determine a random amount to pool this layer by, 2^(0-2)
                                random_pooling_value = np.random.randint(0, 3)
                                # Define the output size of this layer as being the previous layer's output divided
                                # by this pooling amount
                                layerSizes_[depth] = int(layerSizes_[depth-1] / (2 ** random_pooling_value))

                                # Add all parameters together to create a single modelParams dictionary, and add it to
                                # the list
                                modelParams.append({"layerSizes": layerSizes_,
                                                    "filterSizes": filterSizes,
                                                    "finalPool": 2 ** final_pool_size_exp,
                                                    "denseSize": 2 ** dense_size_exp})

# Shuffle the list of generated model parameters
np.random.shuffle(modelParams)


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



'''
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
bestMean = 0
bestBest = 0
modelCount = 0

'''
modelParams[0] = ({"layerSizes": [128, 64, 32],
                   "filterSizes": [4, 8, 16],
                   "finalPool": 8,
                   "denseSize": 512})
                   '''

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
    print("CURRENT BEST = " + str(bestMean))

    # If the new average result is close to or better than the best average result
    if accMean > bestMean*0.95:
        # Append the results and parameters to file
        f = open("bestMean.txt", "a")

        s = "\n--------------------------\n\nAVERAGE = " + str(accMean) + "\nINDIVIDUAL = " + \
            str(accResults) + "\nPARAMS = " + str(modelParam)
        f.write(s)
        f.close()

        # If this was the new best result, replace the old best result
        if accMean > bestMean:
            bestMean = accMean

    # If the new best fold result is close to or better than the best best-fold result
    if accBest >= bestBest*0.95:
        # Append the results and parameters to file
        f = open("bestBest.txt", "a")

        s = "\n--------------------------\n\nBEST = " + str(accBest) + "\nAVERAGE = " + str(accMean) + \
            "\nINDIVIDUAL = " + str(accResults) + "\nPARAMS = " + str(modelParam)
        f.write(s)
        f.close()

        # If this was the new best result, replace the old best result
        if accBest > bestBest:
            bestBest = accBest





'''
model = build_model((DOC_LENGTH, WORD_FEATURES),
                    [128, 64, 32],
                    [FILTER_LENGTH, FILTER_LENGTH*2, FILTER_LENGTH*4],
                    8,
                    512)

av_acc = train_and_test(model, vectorDocs, labels, kFold)

print("AVERAGE ACCURACY = " + str(av_acc))
'''

















