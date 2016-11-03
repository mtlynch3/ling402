#!/usr/bin/python3

import sys
import nltk
import math

from d import Vocabulary
from c import Conditional


class ParallelCorpus:

    # Define a constructor
    def __init__(self):

        # List of English sentences. Each sentence will be represented as a list of ints.
        self.e = list() 

        # List of foreign sentences  Each sentence will be represented as a list of ints.
        self.f = list() 

        # Initially empty vocabularies
        self.e_vocab = Vocabulary()
        self.f_vocab = Vocabulary()


    # Returns the number of sentence pairs that have been added to this parallel corpus
    def size(self):
        return len(self.e)

    # Returns the list of integers corresponding to the English sentence at the specified sentence index
    def get_e(self, sentence_index):
        return self.e[sentence_index]

    # Returns the list of integers corresponding to the foreign sentence at the specified sentence index
    def get_f(self, sentence_index):
        return self.f[sentence_index]


    # Given a string representing an English sentence
    #   and a string representing a foreign sentence,
    #   tokenize each string using nltk.word_tokenize,
    #   and use the appropriate vocabulary to convert each token to an int.
    #   
    # Append the list of integers (corresponding to the English sentence) to self.e
    # Append the list of integers (corresponding to the foreign sentence) to self.f
    def add(self, e, f):
        pass


    # Construct a conditional distribution with the given name.
    #
    # Use the formula given in the supplementary instructions
    def create_uniform_distribution(self, name):
        pass


    # Given a sentence index, a scaling factor epsilon, and a conditional distribution,
    #    calculate the conditional probability 
    #    of the English sentence (at that sentence index) 
    #    given the foreign sentence (at that sentence index)
    #
    # Use the formula given in the supplementary instructions
    def conditional_probability(self, sentence_index, epsilon, conditional):
        pass


    # Given a conditional distribution and a scaling factor epsilon,
    #    calculate the perplexity of this parallel corpus.
    #
    # Use the formula given in the supplementary instructions
    def perplexity(self, epsilon, conditional):
        pass


if __name__ == '__main__':
    
    corpus = ParallelCorpus()

    if len(sys.argv) > 1:
        f = open(sys.argv[1])
        for line in f:
            e,f = line.split("\t")
            corpus.add(e,f)

    else:

        corpus.add("the house", "das Haus")
        corpus.add("the book",  "das Buch")
        corpus.add("a book",    "ein Buch")

        
    epsilon = 0.01
    t = corpus.create_uniform_distribution("t")
    ppl = corpus.perplexity(epsilon, t)
    print(ppl)
