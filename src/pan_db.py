import numpy as np
import itertools
import sys
import os
import shutil
import math
import sqlite3 as lite


class PanDatabaseManager(object):
    def __init__(self, database):
        self.database = database
        db = lite.connect(database)
        self.cur = db.cursor()

    def get_sentences_metadata(self):
        sentences = self.get_sentences()

        sentences_ids = []
        article_ids = []
        author_ids = []
        isplag_flags = []

        for sentence in sentences:
            sentences_ids.append(sentence[0])
            article_ids.append(sentence[1])
            author_ids.append(sentence[2])
            isplag_flags.append(sentence[2])

        # sentences_ids, article_ids, author_ids, isplag_flags = map(
        #     list, zip(self.get_sentences())
        # )

        return sentences_ids, article_ids, author_ids, isplag_flags

    def get_ids_for_documents(self):
        sql = 'SELECT id FROM article order by id'
        self.cur.execute(sql)
        resultset = self.cur.fetchall()
        return [row[0] for row in resultset]

    def get_sentences(self):
        sql = 'SELECT id, fk_article_id, fk_author_id, isplag FROM sentence'
        self.cur.execute(sql)
        return self.cur.fetchall()

    def get_ids_and_contents_of_sentences(self):
        sql = 'SELECT id, fragment FROM sentence'
        self.cur.execute(sql)
        result = self.cur.fetchall()
        ids, sentences = map(list, zip(*result))
        return ids, sentences

    def get_plagiarized_sentences_for_doc(self, doc_id):
        sql = 'SELECT id, offset, length FROM sentence where fk_article_id = :doc_id and isplag = TRUE'

        self.cur.execute(sql, {"doc_id": int(doc_id)})
        result = self.cur.fetchall()
        if len(result) > 0:
            ids, offsets, lengths = map(list, zip(*result))
            return ids, offsets, lengths
        else:
            # handle empty result set
            return [], [], []

    def get_sentences_for_doc(self, doc_id):
        sql = 'SELECT id, offset, length FROM sentence where fk_article_id = :doc_id order by id'
        self.cur.execute(sql, {"doc_id": int(doc_id)})
        result = self.cur.fetchall()
        ids, offsets, lengths = map(list, zip(*result))
        return ids, offsets, lengths

    def get_sentence_texts(self):
        '''
        Get all sentences (i.e., their textual content) contained in the database.
        '''
        sql = 'SELECT fragment FROM sentence'
        self.cur.execute(sql)
        resultset = self.cur.fetchall()
        return [x[0] for x in resultset]

    def get_sentences_texts_for_doc(self, doc_id):
        '''
        Get all sentences (i.e., their textual content) contained in the database for a particular document.
        '''
        sql = 'SELECT fragment FROM sentence where fk_article_id = :doc_id order by id'
        self.cur.execute(sql, {"doc_id": int(doc_id)})
        resultset = self.cur.fetchall()
        return [x[0] for x in resultset]

    def get_offset_and_length_for_sentence(self, sentence_id):
        """
        Retrieve the offset and length of a sentence identified by sentence_id
        """
        sql = 'SELECT offset, length FROM sentence where id = :sent_id'
        self.cur.execute(sql, {"sent_id": int(sentence_id)})
        result = self.cur.fetchall()
        assert len(result) == 1
        return result[0][0], result[0][1]

    def get_filename_for_doc(self, doc_id):
        sql = 'SELECT filename FROM article where id = :doc_id'
        self.cur.execute(sql, {"doc_id": int(doc_id)})
        result = self.cur.fetchone()
        return result[0]
