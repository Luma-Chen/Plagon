import numpy as np
import sys
import pickle
from keras.models import Model
from keras.models import model_from_json
import os
import argparse
from pan_db import PanDatabaseManager

def get_stvecs_for_doc(st_dir, doc_id):
    stvecs_filename = os.path.join(st_dir, 'stvecs{:05d}.pkl'.format(doc_id))
    if not os.path.isfile(stvecs_filename):
        print('File %s not found.' % stvecs_filename) 
        return None
    else:
        with open(stvecs_filename, "rb") as file:
            vecs = pickle.load(file)
        return vecs

def load_model():
    # load json and create model
    json_file = open('../siamese_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../siamese_model/model.h5")
    print("Loaded model from disk")
    return loaded_model

def main():
    print("Going to generate graphs for each test document...")

    # load siamese net model
    loaded_model = load_model()

    parser = argparse.ArgumentParser(description='Generates a set of graphs, one for each test document.')
    parser.add_argument(
        "--pandb",
        help = 'specify the sqlite PAN database file', 
        required=True)
    parser.add_argument(
        "--st_dir", 
        help = 'source directory of the SkipThouths sentence embeddings', 
        required=True)
    parser.add_argument(
        "--dest_dir", 
        help = 'destination directory for the graphs to be generated', 
        required=True)
    parser.add_argument(
        "--tuples_dir", 
        help = 'source directory of the tuples files', 
        required=True)
    parser.add_argument(
        "--max_docs", 
        help = 'max amount of docs to be processed (if not specified, all docs are processed)', 
        required=False,
        type=int)

    args = parser.parse_args()

    # 
    # Now, for each doc, generates its corresponding graph.
    #
    # ***NB*** 
    # The CPLEX-based program that applies the CC requires that the labels of vertices 
    # in a input graph G are integers in the range from 1..n, 
    # n being the amount of vertices in G. Because of this,
    # I could not use the sentence ids as the labels of vertices.
    # To circunvent this problem, I created a mapping from the sentence ids to the 
    # values in 1..n. The mapping for each graph is stored in a pkl file.
    # For example, for the document whose id is 362, the following two files
    # will be created:
    # - doc00362.txt, the graph representation of doc 362
    # - doc00362.pkl, the mapping from sentence ids to local ids (i.e., vertices' labels).
    #
    filelist = os.listdir(args.tuples_dir)
    filelist.sort()

    pandb = PanDatabaseManager(args.pandb)

    num_docs = 0
    for file in filelist:
        filename = os.path.join(args.tuples_dir, file)
        if filename.endswith('.pkl'):

            if args.max_docs is not None:
                if num_docs >= args.max_docs:
                    break

            num_docs = num_docs + 1

            doc_id = int(filename[-9:-4])

            vecs = get_stvecs_for_doc(args.st_dir, doc_id)
            if vecs is None:
                print('Graph for doc %d not built (sentence embeddings not found).' % doc_id)
                continue

            ids, _, _ = pandb.get_sentences_for_doc(doc_id)

            sequential_id = 1
            mapping_from_sent_ids_to_local_ids = dict()

            tuples_file = open(filename, "rb")
            tuples = pickle.load(tuples_file)

            info_for_curr_doc = []
            X_list = []

            max_dist = 0.0
            for cur_tuple in tuples:
                doc_id = cur_tuple[0]                
                sentence_id1 = cur_tuple[1]
                sentence_id2 = cur_tuple[2]
                similarity_bit = cur_tuple[3]

                idx1 = ids.index(sentence_id1)
                idx2 = ids.index(sentence_id2)

                # load the sentence embedding vectors for each sentence.
                vec1, vec2 = vecs[idx1], vecs[idx2]
                example = (vec1.detach().numpy(), vec2.detach().numpy())

                X_list.append(example)

                if sentence_id1 in mapping_from_sent_ids_to_local_ids:
                    local_sentence_id1 = mapping_from_sent_ids_to_local_ids[sentence_id1]
                else:
                    mapping_from_sent_ids_to_local_ids[sentence_id1] = sequential_id 
                    local_sentence_id1 = mapping_from_sent_ids_to_local_ids[sentence_id1]
                    sequential_id = sequential_id + 1

                if sentence_id2 in mapping_from_sent_ids_to_local_ids:
                    local_sentence_id2 = mapping_from_sent_ids_to_local_ids[sentence_id2]
                else:
                    mapping_from_sent_ids_to_local_ids[sentence_id2] = sequential_id 
                    local_sentence_id2 = mapping_from_sent_ids_to_local_ids[sentence_id2]
                    sequential_id = sequential_id + 1

                # NB: we store the similarity_bit in order to validate the quality of the clustering.
                info_for_curr_doc.append((local_sentence_id1, local_sentence_id2, similarity_bit))

            tuples_file.close()

            te_pairs = np.array(X_list)

            dist = loaded_model.predict([te_pairs[:, 0], te_pairs[:, 1]])

            max_dist = np.amax(dist, axis=0)
            print('max_dist: ', max_dist)

            num_edges = len(info_for_curr_doc)
            num_vertices = len(mapping_from_sent_ids_to_local_ids)

            print('Generating graph for document %d. max_dist: %f, #vertices: %d, #edges: %d' % 
                (doc_id, max_dist, num_vertices, num_edges))

            #
            # Now that all relevant info for current doc is collected, we proceed to build
            # the corresponding graph. Such a graph will be stored in a text file in the 
            # format expected by the program implementing the Correlation Clustering algorithm.
            #
            txt_filename = os.path.join(args.dest_dir, "doc{:05d}.txt".format(doc_id))
            f = open(txt_filename, "w")

            # First write the lines corresponding to the header
            f.write("%d\r\n%d\r\n%d\r\n%.2f\r\n" % (0, 0, num_vertices, 0.0))

            # Now write info about the edges to the txt file
            i = 0
            for sent_pair_info in info_for_curr_doc:
                local_sentence_id1 = sent_pair_info[0]
                local_sentence_id2 = sent_pair_info[1]
                normalized_dist = (dist[i]/max_dist)    # value in the range [0,1]
                similarity = 1 - normalized_dist        # maps distance to similarity
                similarity = similarity - .5            # maps to the range [-0.5, +0.5]
                f.write("%d %d %.2f\r\n" % (local_sentence_id1, local_sentence_id2, similarity))
                i = i + 1

            f.close()

        # Write info about the mapping (sentence id --> local id) to a pkl file
        pkl_filename = os.path.join(args.dest_dir, "doc{:05d}.pkl".format(doc_id))
        pkl_file = open(pkl_filename, "wb")
        pickle.dump(mapping_from_sent_ids_to_local_ids, pkl_file)
        pkl_file.close()

    print('Done!')

'''
    Execution examples:
        python gen_graphs.py --pandb plag_train.db --tuples_dir ../tuples --st_dir ./stvecs_by_doc --max_docs 2355 --desti_dir "../graphs"
        python gen_graphs.py --pandb plag_train.db --tuples_dir ../tuplesULTD --st_dir ./stvecs_by_doc --dest_dir "../graphsULTD"
'''
if __name__ == "__main__":
    main()
