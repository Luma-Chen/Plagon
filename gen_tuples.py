# coding=utf-8
import numpy as np
import sys
import pickle
import argparse, os

# carrega o arquivo com metadados ordenados por author_ids, isplag_flags, sentences_ids
def load_metadata():
    dbfile = open('./data/pkl/metadata.pkl', 'rb')
    metadata = pickle.load(dbfile)
    return metadata

def generate_tuples_limited(doc_id, ids_of_original_sentences, ids_of_plagiarized_sentences, maxplag):
    max_original = min(len(ids_of_original_sentences), maxplag)
    max_plagiarized = min(len(ids_of_plagiarized_sentences), maxplag)

    tuples = []

    #
    # Generate negative (i.e., dissimilar sentences) examples
    #
    qty_neg = 0
    for i in range(max_original):
        for j in range(max_plagiarized):
            training_tuple = (doc_id, ids_of_original_sentences[i], ids_of_plagiarized_sentences[j], 0)
            tuples.append(training_tuple)
            qty_neg = qty_neg + 1

    #
    # Generate positive (i.e., similar sentences) examples
    #
    stop_flag = False
    qty_pos = 0
    for i in range(len(ids_of_original_sentences)):
        if stop_flag:
            break
        for j in range(i+1,len(ids_of_original_sentences)):
            training_tuple = (doc_id,ids_of_original_sentences[i],ids_of_original_sentences[j],1)
            tuples.append(training_tuple)
            qty_pos = qty_pos + 1
            if qty_pos >= qty_neg:
                stop_flag = True
                break

    return tuples, qty_neg, qty_pos

def generate_tuples_unlimited(doc_id, 
                ids_of_original_sentences, 
                ids_of_plagiarized_sentences):
    tuples = []

    #
    # Generate negative (i.e., dissimilar sentences) examples
    #
    qty_neg = 0
    for i in range(len(ids_of_original_sentences)):
        for j in range(len(ids_of_plagiarized_sentences)):
            training_tuple = (doc_id, ids_of_original_sentences[i], ids_of_plagiarized_sentences[j], 0)
            tuples.append(training_tuple)
            qty_neg = qty_neg + 1

    #
    # Generate neutral examples
    #
    qty_neutral = 0
    for i in range(len(ids_of_plagiarized_sentences)):
        for j in range(i+1,len(ids_of_plagiarized_sentences)):
            training_tuple = (doc_id,ids_of_plagiarized_sentences[i],ids_of_plagiarized_sentences[j],0.5)
            tuples.append(training_tuple)
            qty_neutral = qty_neutral + 1

    #
    # Generate positive (i.e., similar sentences) examples
    #
    qty_pos = 0
    for i in range(len(ids_of_original_sentences)):
        for j in range(i+1,len(ids_of_original_sentences)):
            training_tuple = (doc_id,ids_of_original_sentences[i],ids_of_original_sentences[j],1)
            tuples.append(training_tuple)
            qty_pos = qty_pos + 1

    return tuples, qty_neg, qty_pos, qty_neutral
    

def main():
    parser = argparse.ArgumentParser(description='Generates intermediate data structure to be used to build training examples.')
    parser.add_argument(
        "--maxplag", 
        type=int,
        help = 'maximum amount of plagiarized sentences to be used in each document', 
        required=False)
    parser.add_argument(
        "--minsent", 
        type=int,
        help = 'minimum amount of sentences for a document to be considered', 
        required=False)
    parser.add_argument(
        "--maxsent", 
        type=int,
        help = 'maximum amount of sentences for a document to be considered', 
        required=False)
    parser.add_argument(
        "--destdir", 
        help = 'destination directory for tuple files (one pkl file per document)', 
        required=True)

    args = parser.parse_args()

    if args.maxplag is None:
        print("Argument 'maxplag' not specified. So, considering all possible combinations of sentences per doc.")
    else:
        print("Argument 'maxplag' set to %d." % args.maxplag)

    metadata = load_metadata()
    document_ids = metadata['article_ids']
    sentences_ids = metadata['sentences_ids']
    isplag_flags = metadata['isplag_flags']

    print("Amount of sentence: ", len(sentences_ids))

    num_doc = 0
    num_autor = 0
    similarity =''

    total_qty_pos = total_qty_neg = 0
    document_ids_set = sorted(set(document_ids))

    qty_ignored_docs = 0

    print('Total docs: %d.' % len(document_ids_set))

    for doc_id in document_ids_set:

        tuples_filename = os.path.join(args.destdir, "tuples{:05d}.pkl".format(doc_id))
        if os.path.isfile(tuples_filename):
            print('File for doc %d has already been created; moving to the next one' % doc_id)
            continue

        # get indices of all entries associated to the current document
        indices = list(filter(lambda x: document_ids[x] == doc_id, range(len(document_ids))))

        # get all sentences associated to the current document, along with their plag flag.
        sentences_list = list(np.array(sentences_ids)[indices])
        flags_list = list(np.array(isplag_flags)[indices])

        indices_original_sentences = list(filter(lambda x: flags_list[x] == 0, range(len(flags_list))))
        indices_plagiarized_sentences = list(filter(lambda x: flags_list[x] == 1, range(len(flags_list))))

        print('Document %d has %d plagiarized sentences, and %d original sentences' % 
            (doc_id,
                len(indices_plagiarized_sentences), 
                len(indices_original_sentences)))

        # documents with 0 original or 0 plagiarized sentences are ignored.
        n_plag = len(indices_plagiarized_sentences)
        n_orig = len(indices_original_sentences)
        if (n_orig == 0) or (n_plag == 0):
            print('Document %d ignored (n_orig: %d, n_plag: %d)' % (doc_id, n_orig, n_plag))
            qty_ignored_docs = qty_ignored_docs + 1
            continue

        ids_of_original_sentences = list(np.array(sentences_list)[indices_original_sentences])
        ids_of_plagiarized_sentences = list(np.array(sentences_list)[indices_plagiarized_sentences])

        n = n_orig + n_plag

        if args.minsent is not None:
            if n < args.minsent:
                print('Document %d ignored (too few sentences: %d)' % (doc_id, n))
                continue

        if args.maxsent is not None:
            if n > args.maxsent:
                print('Document %d ignored (too many sentences: %d)' % (doc_id, n))
                continue

        tuples_file = open(tuples_filename, "wb")

        if args.maxplag is None:
            tuples, qty_neg, qty_pos, qty_neutral = generate_tuples_unlimited(doc_id, 
                ids_of_original_sentences, 
                ids_of_plagiarized_sentences)
            print('\tGenerated %d pos tuples (similar sentence pairs).' % qty_pos)
            print('\tGenerated %d neg tuples (dissimilar sentence pairs).' % qty_neg)
            print('\tGenerated %d neutral tuples (plag_x_plag sentence pairs).' % qty_neutral)
        else:
            tuples, qty_neg, qty_pos = generate_tuples_limited(doc_id, 
                ids_of_original_sentences, 
                ids_of_plagiarized_sentences, 
                args.maxplag)
            print('\tGenerated %d pos tuples (orig_x_orig sentence pairs).' % qty_pos)
            print('\tGenerated %d neg tuples (orig_x_plag sentence pairs).' % qty_neg)

        total_qty_pos = total_qty_pos + qty_pos
        total_qty_neg = total_qty_neg + qty_neg

        pickle.dump(tuples, tuples_file)

        tuples_file.close()

    print("Done!")
    print('Total amount of pos generated tuples: %d.' % total_qty_pos)                   
    print('Total amount of pos generated tuples: %d.' % total_qty_neg) 
    print('Total amount of ignored docs: %d.' % qty_ignored_docs)                   

'''
    This program generates a intermediate structure (stored as a pickle file tuples.pkl) 
    that is latter used to generate the training examples
    for fitting the siamese neural network model. 

    This intermediate structure comprises a sequence of tuples
    in the form (doc_id, s1_id, s2_id, flag), where:
        - doc_id is a identifier for a document d;
        - s1_id and s2_id are identifiers of two sentences inside document d; 
        - flag is an indicator (i.e., a binary value): 1 if s1 and s2 are similar, 0 otherwise. 
    
    Two sentences are considered similar iff they both do not correspond to plagiarism.

    The program scans each document. For each document, its plagiarized and original sentences are retrieved.
    Then, these two lists used to form the tuples.

    NB: case 1 (argmax is specified) the argument --maxplag specifies the maximum amount N of plagiarized 
        sentence to be considered in a given document. if N is lesser than the amount of plagiarized sentences
        in a document d, only the first N plagiarized sentences os d are used to generate tuples.
        The program also sees to generate balanced amounts of similar and dissimilar tuples for each document.

        case 2 (argmax is not specified): all possible combinations of sentences in each doc are considered.


    Execution examples:
        python gen_tuples.py --maxplag 40 --destdir ../tuples40
        python gen_tuples.py --destdir ../tuplesULTD --maxsent 50
        python gen_tuples.py --destdir ../tuplesULTD_51_100 --minsent 51 --maxsent 100
'''
if __name__ == "__main__":
    main()
