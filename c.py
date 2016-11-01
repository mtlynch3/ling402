#!/usr/bin/python3

import sys
import nltk

# Import d.py
from d import Vocabulary

class Conditional:

    # Define a constructor
    #
    # For every pair of words (one from e, one from f),
    #    store the value provided by initial_value
    # 
    # HINT: In addition to storing the name and both vocabs,
    #       you will need to store a dictionary of dictionaries
    def __init__(self, name, e_vocab, f_vocab, initial_value):
        pass

    # Given an integer index for a word from e and a word from f,
    #    return the corresponding value
    def get(self, e_i, f_i):
        pass

    # Given an integer index for a word from e and a word from f,
    #    store the value provided
    def set(self, e_i, f_i, value):
        pass

    # Return a string representation of this object
    #
    # See c.expected_output and test_c.py for the format of the string
    def __str__(self):
        pass


def create_vocab(words):
    v = Vocabulary()
    for word in words:
        v.get_int(word)
    return v


if __name__ == '__main__':

    if (len(sys.argv) > 2):
        f1 = open(sys.argv[1])
        raw1 = f1.read()
        words_e=nltk.word_tokenize(raw1)

        f2 = open(sys.argv[2])
        raw2 = f2.read()
        words_f=nltk.word_tokenize(raw2)

    else:
        words_e = nltk.word_tokenize("All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.")
        
        words_f = nltk.word_tokenize("Todos los seres humanos nacen libres e iguales en dignidad y derechos y, dotados como están de razón y conciencia, deben comportarse fraternalmente los unos con los otros.")

        e_v = create_vocab(words_e)
        f_v = create_vocab(words_f)
        
        count = Conditional("count", e_v, f_v, 0)
        print(count)
        print()
