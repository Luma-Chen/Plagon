import numpy as np
import sys
from scipy import spatial
import pickle
import argparse
import os
from pan_db import PanDatabaseManager

def get_stvecs_for_doc(doc_id):
    stvecs_filename = './stvecs_by_doc/stvecs{:05d}.pkl'.format(doc_id)
    if not os.path.isfile(stvecs_filename):
        print('File %s not found.' % stvecs_filename) 
        return None
    else:
        with open(stvecs_filename, "rb") as file:
            vecs = pickle.load(file)
        return vecs

def main():
    total_qty = 0
    total_qty_pos = 0

    parser = argparse.ArgumentParser(description='Generates examples to train the siamese net.')
    parser.add_argument(
        "--pandb",
        help = 'specify the sqlite PAN database file', 
        required=True)
    parser.add_argument(
        "--srcdir", 
        help = 'source directory of tuples files', 
        required=True)
    parser.add_argument(
        "--destdir", 
        help = 'destination directory for the triple files (one pkl per document)', 
        required=True)
    parser.add_argument(
        '--start', 
        type=int,
        help = 'doc id at which the processing should start', 
        required = True)
    args = parser.parse_args()

    pandb = PanDatabaseManager(args.pandb)
    ids_for_docs = pandb.get_ids_for_documents()

    document_ids_set = sorted(set(ids_for_docs))

    ignored_tuples = 0
    
    for doc_id in document_ids_set:
        if doc_id < args.start:
            continue

        vecs = get_stvecs_for_doc(doc_id)
        if vecs is None:
            continue

        examples = []

        tuples_filename = os.path.join(args.srcdir, "tuples{:05d}.pkl".format(doc_id))
        if not os.path.isfile(tuples_filename):
            print('File %s not found.' % tuples_filename) 
            continue
        else:
            tuples_file = open(tuples_filename, "rb")

        triples_filename = os.path.join(args.destdir, "triples{:05d}.pkl".format(doc_id))
        triples_file = open(triples_filename, "wb")

        tuples = pickle.load(tuples_file)

        print('Generating training examples for doc %d' % doc_id)
        
        ids, _, _ = pandb.get_sentences_for_doc(doc_id)

        for cur_tuple in tuples:
            sentence_id1 = cur_tuple[1]
            sentence_id2 = cur_tuple[2]
            similarity_bit = cur_tuple[3]

            if similarity_bit == 0.5:
                ignored_tuples = ignored_tuples + 1
                continue

            idx1 = ids.index(sentence_id1)
            idx2 = ids.index(sentence_id2)

            vec1, vec2 = vecs[idx1], vecs[idx2]
            example = (vec1.detach().numpy(), vec2.detach().numpy(), similarity_bit)

            examples.append(example)

        pickle.dump(examples, triples_file)

        tuples_file.close()
        triples_file.close()

        total_qty = total_qty + 1
        total_qty_pos = total_qty_pos + similarity_bit

    print('Done!')
    print('Amount of generated examples: %d.' % total_qty)
    print('Amount of positive examples: %d.' % total_qty_pos)
    print('Amount of negative examples: %d.' % (total_qty-total_qty_pos))
    print('Amount of ignored (neutral) tuples: %d.' % ignored_tuples)

'''
    Generate triples (examples) to be latter used to fit the the siamese neural network model.

    NB: a file of tuples may contain entries in which the similarity_bit is 0.5. These
        entries corresponds to combinations of sentences plag_x_plag. These entries are 
        not used to make training triples.

    Execution examples:
        gen_triples.py --pandb plag_train.db --srcdir ../tuples40 --destdir ../triples40 --start 1
'''
if __name__ == "__main__":
    main()
