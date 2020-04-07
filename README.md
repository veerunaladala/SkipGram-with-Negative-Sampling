To start with, We divided the problem statement into modules, with each module covering important aspects of the skip-gram model.

# Preprocessing:

First, we tried Spacy preprocessing but later for convenience, we used traditional
preprocessing technique of removing punctuations, splitting on space, converting to lower
case and removing one-word sentences.

               text -> sentences -> clean the sentences -> tokenization

# Encoding the word vectors

In our initial trials, to improve the computational efficiency we tried to encode the words in
a way that’s different from one hot encoding - making a list of lists with each list in the main
list starting with center word followed by context words. But moving ahead with the code
we found it difficult for us to play around with those vectors when calculating gradient and
updating weights. Therefore, we finally used one hot vector encoding to encode the words
in the corpus.

# Negative Sampling

Initially we tried picking the samples just randomly from the vocab(omitting center word
and context word). After reading a few papers on the negative sampling, we understood
that negative sampling is widely done using a noise distribution. However, the unigram
distribution is raised to the power of 3/4rd to combat the imbalance between common vs
rare words. Hence we finally defined a function such that it generates the noise distribution
and returns the words with their probability. The Sample function then uses these
probabilities to pick 5 samples and returns them.

# Neural Network Implementation detail

This is the most important part of the entire code. We have extensively read various papers
and We checked the math part of this project regarding to forward prop, loss calculation,
and backward prop.

We initially started off our trials with implementing just the skipgram model without
negative sampling to clearly understand the working of forward and backward propagation
step by step, where in we defined different functions for forward and backward passes.

But when the negative sampling is included, the objective functions for context words and
negative samples are different. Therefore, after calculating the hidden layer, we eliminated
the back propagation function from our skipgram model and separately performed the
stochastic gradient descent and calculated the gradients for context words and negative words and updated the center word vector in input weight matrix and K+1 (negative +
context) word vectors in output weight matrix.


# Similarity
We used Cosine Similarity to calculate the similarity between two vector embeddings.
