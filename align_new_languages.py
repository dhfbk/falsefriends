import numpy as np
from fasttext import FastVector
import argparse
import re

"""
Part of this code is taken from:
	https://github.com/Babylonpartners/fastText_multilingual
"""

parser = argparse.ArgumentParser()
parser.add_argument("lang1", help="Language 1 (modified)")
parser.add_argument("lang2", help="Language 2 (not modified)")
parser.add_argument("out1", help="Output vectors for lang1")
parser.add_argument("out2", help="Output vectors for lang2")
parser.add_argument("dict", help="Dictionary used to align the languages (format: lang1 [tab] lang2)")
args = parser.parse_args()

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

lang1_dictionary = FastVector(vector_file=args.lang1)
lang2_dictionary = FastVector(vector_file=args.lang2)

bilingual_dictionary=[]
file_object  = open(args.dict, "r")
lines = file_object.readlines()
for line in lines:
    line = re.sub(r'\n', '', line)
    w_lang2, w_lang1 = line.split('\t')
    if w_lang1 in lang1_dictionary.word2id.keys() and w_lang2 in lang2_dictionary.word2id.keys():
        bilingual_dictionary.append(tuple((w_lang2, w_lang1)))

print("Dic Size: " + str(len(bilingual_dictionary)))

# form the training matrices# form  
source_matrix, target_matrix = make_training_matrices(lang1_dictionary, lang2_dictionary, bilingual_dictionary)
# learn and apply the transformation
transform = learn_transformation(source_matrix, target_matrix)
lang1_dictionary.apply_transform(transform)

lang1_dictionary.export(args.out1)
lang2_dictionary.export(args.out2)
