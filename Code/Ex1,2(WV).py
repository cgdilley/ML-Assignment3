import os
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense

#

# DEFINE CONSTANTS
# ==================

#

# Number of features for the word vectors (should be 50, 100, 200, or 300)
WORDVECTOR_FEATURES = 300
# Number of folds for k-fold cross-validation
K_FOLDS = 10
# Number of hidden dimensions for MLP
HIDDEN_DIM = 1024
# Parameters for Logistic Regression
LOGREG_C = 50
LOGREG_PENALTY = "l1"
# Keys to identify particular fields by
FILENAME_KEY = 'filename'
BIGRAMS_KEY = 'bigrams'
UNIGRAMS_KEY = 'unigrams'
LABEL_KEY = '%LABEL%'
BIGRAM_VECTOR_SUM_KEY = "bigramVectorAverage"
UNIGRAM_VECTOR_SUM_KEY = "unigramVectorAverage"
# Special strings to serve and tokens at the beginning and end of sentences
SENT_START = "<s>"
SENT_END = "</s>"
# The minimum frequency a token (or bigram) must have in order to be considered a feature
FREQ_THRESHOLD = 5

#

# LOAD WORD VECTOR DATA
# ======================

#

print("Loading word vectors...")

wordVectors = {}

# Open the word vector file
vectorFile = open("./glove.6B/glove.6B.XXd.txt".replace("XX", str(WORDVECTOR_FEATURES)), 'r')
# Split by newline into individual words
words = vectorFile.read().split("\n")
# Put the word along with its feature vector into dictionary
for word in words:
    vals = word.split(" ")
    if vals[0] != '':
        wordVectors[vals[0]] = [float(feat) for feat in vals[1:]]

#

# READ AND PROCESS DATA
# ======================

#

# Initialize key-value structures that will count the frequency of all unigrams and bigrams in the entire corpus
unigramTypeMap = {}
bigramTypeMap = {}

# Initialize lists of all documents, keeping track of their filename, label, and relative frequencies
documents = []

print("Processing files...")


def increment_dict(dict_, field):
    # Function for incrementing a counter in a map
    # dict_: Dictionary to reference
    # field: Key name of field to increment

    if field in dict_:
        dict_[field] += 1
    else:
        dict_[field] = 1
    return


# Iterate through all files in the corpus
for filename in glob.glob("./review_polarity/txt_sentoken/*/*.txt"):
    # Open the file
    with open(filename, 'r') as file:
        # Read the data in the file as a string
        data = file.read()
        # Split the data into a list of sentences
        sentences = data.split('\n')

        # Create new map to represent this document to add to the document list once completed
        label = 0 if 'neg' in filename else 1
        thisDoc = {FILENAME_KEY: filename,
                   LABEL_KEY: label,
                   UNIGRAMS_KEY: {},
                   BIGRAMS_KEY: {},
                   UNIGRAM_VECTOR_SUM_KEY: [0.0] * WORDVECTOR_FEATURES,
                   BIGRAM_VECTOR_SUM_KEY: [0.0] * (WORDVECTOR_FEATURES * 2)}

        # Initialize counters for total number of unigrams and bigrams in the text
        unigramCount = 0
        bigramCount = 0
        for sentence in sentences:
            # Split the data into a list of tokens
            all_tokens = sentence.split(' ')
            # Remove empty tokens
            tokens = [token for token in all_tokens if token != '']

            # Iterate through all tokens
            for i in range(len(tokens)):
                token = tokens[i]

                # Increment unigram counter
                unigramCount += 1

                # Increment counter for the token in the corpus's main map
                increment_dict(unigramTypeMap, token)

                # Increment counter for the token in the document's map
                increment_dict(thisDoc[UNIGRAMS_KEY], token)

                # Add the word vector for this token to the word vector sum of this document
                if token in wordVectors:
                    thisDoc[UNIGRAM_VECTOR_SUM_KEY] = \
                        [x + y for x, y in zip(wordVectors[token], thisDoc[UNIGRAM_VECTOR_SUM_KEY])]

                # If we're not looking at the last token, count unique bigrams of this token and the following
                if i < len(tokens) - 1 and tokens[i + 1] != '':
                    bigram = tokens[i] + "_" + tokens[i + 1]

                    # Increment bigram counter
                    bigramCount += 1

                    # Increment counter for the bigram in the corpus's main map
                    increment_dict(bigramTypeMap, bigram)

                    # Increment counter for the bigram in the document's map
                    increment_dict(thisDoc[BIGRAMS_KEY], bigram)

                    # Add the bigram vector for this bigram to the bigram vector sum of this document
                    if tokens[i] in wordVectors and tokens[i + 1] in wordVectors:
                        bigramVector = wordVectors[tokens[i]] + wordVectors[tokens[i + 1]]
                        thisDoc[BIGRAM_VECTOR_SUM_KEY] = \
                            [x + y for x, y in zip(bigramVector, thisDoc[BIGRAM_VECTOR_SUM_KEY])]

            # Add special bigrams for the beginning and end of sentences
            # If the list of tokens is non-empty
            if len(tokens) > 0:
                # Get first and last tokens
                firstToken = tokens[0]
                lastToken = tokens[len(tokens) - 1]

                # Create starting and ending bigrams using sentence start and end tokens
                startBigram = SENT_START + "_" + firstToken
                endBigram = lastToken + "_" + SENT_END

                # Increment counter for these bigrams in corpus's main map and the document's map
                increment_dict(bigramTypeMap, startBigram)
                increment_dict(bigramTypeMap, endBigram)
                increment_dict(thisDoc[BIGRAMS_KEY], startBigram)
                increment_dict(thisDoc[BIGRAMS_KEY], endBigram)
                bigramCount += 2

        # Adjust unigram and bigram counts to be relative rather than absolute
        for unigram in thisDoc[UNIGRAMS_KEY].keys():
            thisDoc[UNIGRAMS_KEY][unigram] /= unigramCount
        for bigram in thisDoc[BIGRAMS_KEY].keys():
            thisDoc[BIGRAMS_KEY][bigram] /= bigramCount
        thisDoc[UNIGRAM_VECTOR_SUM_KEY] = [feat / unigramCount for feat in thisDoc[UNIGRAM_VECTOR_SUM_KEY]]
        thisDoc[BIGRAM_VECTOR_SUM_KEY] = [feat / bigramCount for feat in thisDoc[BIGRAM_VECTOR_SUM_KEY]]

        # Append this document to the list of documents
        documents.append(thisDoc)
#

#

# FILTER FEATURES BY FREQUENCY
# ==================================

#

print("Filtering features...")

# Initialize the set of all features to use
unigramTypeSet = set()
# Iterate through all types in the map
for key in unigramTypeMap.keys():
    # If the type has a minimum frequency of FREQ_THRESHOLD, add it to the set of features
    if unigramTypeMap[key] >= FREQ_THRESHOLD:
        unigramTypeSet |= {key}

bigramTypeSet = set()
# Iterate through all types in the map
for key in bigramTypeMap.keys():
    # If the type has a minimum frequency of FREQ_THRESHOLD, add it to the set of features
    if bigramTypeMap[key] >= FREQ_THRESHOLD:
        bigramTypeSet |= {key}

# Remove irrelevant features from the documents.
for document in documents:
    document[UNIGRAMS_KEY] = {k: v for k, v in document[UNIGRAMS_KEY].items() if k in unigramTypeSet}
    document[BIGRAMS_KEY] = {k: v for k, v in document[BIGRAMS_KEY].items() if k in bigramTypeSet}

#

#

# BUILD MATRICES FOR LOGISTIC REGRESSION
# =======================================

#



print("Building matrices...")

# Initialize fields and lists for info dictionary.
# Will contain column headers as keys, and lists of row values as values.
# Initializes all values as 0
docLen = len(documents)
unigramInfoDict = {LABEL_KEY: [0] * docLen}
for unigram in unigramTypeSet:
    unigramInfoDict[unigram] = [0] * docLen
bigramInfoDict = {LABEL_KEY: [0] * docLen}
for bigram in bigramTypeSet:
    bigramInfoDict[bigram] = [0] * docLen

# Iterate through all documents
for i in range(len(documents)):
    document = documents[i]

    # Append its label to the label list
    unigramInfoDict[LABEL_KEY][i] = document[LABEL_KEY]
    bigramInfoDict[LABEL_KEY][i] = document[LABEL_KEY]

    # Iterate through all unigram types in the document and insert their values
    for unigram in document[UNIGRAMS_KEY]:
        unigramInfoDict[unigram][i] = document[UNIGRAMS_KEY][unigram]

    # Iterate through all bigram types in the document and insert their values
    for bigram in document[BIGRAMS_KEY]:
        bigramInfoDict[bigram][i] = document[BIGRAMS_KEY][bigram]

# Convert this dictionary of labelled columns and lists of rows into numpy arrays
predictors_uni = np.array([[unigramInfoDict[col][row] for col in unigramTypeSet] for row in range(len(documents))])
labels_uni = np.array([unigramInfoDict[LABEL_KEY][row] for row in range(len(documents))])
predictors_bi = np.array([[bigramInfoDict[col][row] for col in bigramTypeSet] for row in range(len(documents))])
labels_bi = np.array([bigramInfoDict[LABEL_KEY][row] for row in range(len(documents))])

#

#

# ====================================================
# EXERCISE 1:  TRAIN A LOGISTIC REGRESSION CLASSIFIER
# ====================================================

#
print()
print("EXERCISE 1:  Logistic Regression Classifier")
print("============================================")
print("Training and testing model...")
print(" - Unigrams:")
print("-------------")

# Build logistic regression model
m = LogisticRegression(penalty=LOGREG_PENALTY, C=LOGREG_C)

# Test with k-fold cross validation on unigrams, get accuracy scores
ex1_score_uni = cross_val_score(m, predictors_uni, labels_uni, cv=K_FOLDS)

# Output results
print("Results=")
print(ex1_score_uni)
print("MEAN = " + str(ex1_score_uni.mean()))

print()
print(" - Bigrams:")
print("------------")

# Test with k-fold cross validation on bigrams, get accuracy scores
ex1_score_bi = cross_val_score(m, predictors_bi, labels_bi, cv=K_FOLDS)

# Output results
print("Results=")
print(ex1_score_bi)
print("MEAN = " + str(ex1_score_bi.mean()))

#

#

# ====================================================
# EXERCISE 2:  TRAIN A MULTI-LAYER PERCEPTRON
# ====================================================

#

print()
print("EXERCISE 2:  Multi-layer Perceptron")
print("====================================")
print("Converting feature vectors...")

# Build new predictor matrices based on word/bigram feature vectors
predictors_uni_mlp = np.array([[documents[doc][UNIGRAM_VECTOR_SUM_KEY][feat] for feat in range(WORDVECTOR_FEATURES)]
                               for doc in range(len(documents))])
predictors_bi_mlp = np.array([[documents[doc][BIGRAM_VECTOR_SUM_KEY][feat] for feat in range(WORDVECTOR_FEATURES * 2)]
                              for doc in range(len(documents))])

print("Training and testing model...")
print(" - Unigrams:")
print("-------------")

# Construct k-folds
stratFolds = StratifiedKFold(labels_uni, K_FOLDS)


def build_mlp(input_dim, hidden_dim):
    """
    Creates a multi-layer perceptron with the given parameters

    :param input_dim: number of input dimensions (features)
    :param hidden_dim: number of hidden dimensions between layers of MLP
    :return: The constructed mlp model
    """

    # Initialize MLP
    mlp = Sequential()
    # Add dense layer with the given input and output dimensions
    mlp.add(Dense(input_dim=input_dim, output_dim=hidden_dim, activation='relu'))
    # Add second dense layer with single output
    mlp.add(Dense(output_dim=1, activation='sigmoid'))
    # Compile the model
    mlp.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return mlp


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

    return np.array(fold_acc)

# Construct unigram MLP model and train/test by cross-validation
mlp_uni = build_mlp(WORDVECTOR_FEATURES, HIDDEN_DIM)
mlp_uni_acc = train_and_test(mlp_uni, predictors_uni_mlp, labels_uni, stratFolds)

print()
print("Unigram results:")
print(mlp_uni_acc)
print("MEAN = " + str(mlp_uni_acc.mean()))

print()
print(" - Bigrams:")
print("------------")

# Construct bigram MLP model and train/test by cross-validation
mlp_bi = build_mlp(WORDVECTOR_FEATURES*2, HIDDEN_DIM)
mlp_bi_acc = train_and_test(mlp_bi, predictors_bi_mlp, labels_bi, stratFolds)

print()
print("Bigram results:")
print(mlp_bi_acc)
print("MEAN = " + str(mlp_bi_acc.mean()))



