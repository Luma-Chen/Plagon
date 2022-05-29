import numpy as np
import sys
from scipy import spatial
import sqlite3 as lite
import pickle
from itertools import islice
from pan_db import PanDatabaseManager

def load_metadata():
    dbfile = open('./data/pkl/metadata.pkl', 'rb')
    metadata = pickle.load(dbfile)
    return metadata

if __name__ == "__main__":
    pandb = PanDatabaseManager("plag_train.db")
    sentences_ids, article_ids, author_ids, isplag_flags = pandb.get_sentences_metadata()

    print("Amount of sentence ids: ", len(sentences_ids))

    #Saving metadata to pickle file...
    dbfile = open('./data/pkl/metadata.pkl', 'wb')
    metadata_dict = {'sentences_ids': sentences_ids, 'article_ids': article_ids, 'author_ids': author_ids, 'isplag_flags': isplag_flags}
    pickle.dump(metadata_dict, dbfile)
    dbfile.close()

    print("Done!")

    metadata = load_metadata()
    article_ids = metadata['article_ids']
    print(article_ids[:200])

    metadata = load_metadata()
    sentences_ids = metadata['sentences_ids']
    print(sentences_ids[:10])
