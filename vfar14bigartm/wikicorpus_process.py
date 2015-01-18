# -*- coding: utf-8 -*-
import logging, codecs
from gensim.corpora import WikiCorpus
from collections import defaultdict, Counter
import artm.messages_pb2, artm.library 
import os
import glob

#===============================================================================
#         #for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
#             #for tokens, title, pageid in pool.imap(process_article, group): # chunksize=10):
#                 ... // continue with processing tokens
# =>
#         for text in texts:
#                 tokens, title, pageid = process_article(text) # chunksize=10):
#                 ... // continue with processing tokens
#===============================================================================


def load_doc_sets(csv_path='ru2en.csv'):
    doc_set_ru = set()
    doc_set_en = set()
    with open(csv_path, 'r') as csv_file:
        for line in csv_file:
            (id_ru, title_ru, id_en, title_en) = line.split('|') 
            doc_set_ru.add(id_ru)
            doc_set_en.add(id_en)
    print 'Doc set done'
    return (doc_set_ru, doc_set_en)


def save_to_batches(input, doc_set=set(), batch_path='.', batch_size=1000, lang='@body'):
    if not doc_set: # is empty
        return
    wiki = WikiCorpus(input, lemmatize=False, dictionary='empty dictionary')
    wiki.metadata = True  # request to extract page_id and title
    
    num_docs_total = len(doc_set)
    num_docs_batch = 0
    num_docs_curr  = 0
    batch_dict     = defaultdict(int)
    batch_token_id = 0
    NNZ            = 0
    batch = artm.messages_pb2.Batch()
    for (text, page_id_and_title) in wiki.get_texts():
        page_id = page_id_and_title[0]
        title = page_id_and_title[1]
#         print page_id
        
        if page_id in doc_set:
            print num_docs_curr, page_id, title
            # get tokens tf in the text
            text_tf = Counter(text)
            for token in text:
                # update batch dictionary
                if token not in batch_dict:
                    batch_dict[token] = batch_token_id
                    batch_token_id += 1
                    
            # add item to batch
            item = batch.item.add()
            item.id = int(page_id)
            item.title = title
            field = item.field.add()
            field.name = lang
            for token in text_tf:
                field.token_id.append(batch_dict[token])
                field.token_count.append(text_tf[token])
                NNZ += text_tf[token]
       
            num_docs_batch += 1
            num_docs_curr  += 1
            if (num_docs_batch == batch_size) or\
               (num_docs_curr == num_docs_total and num_docs_batch <> 0):
                for token in batch_dict: 
                    batch.token.append(unicode(token, 'utf-8'))
                artm.library.Library().SaveBatch(batch, batch_path)

                print 'Batch done, |W| = ' + str(len(batch.token)) + ", NNZ = " + str(NNZ)

                batch = artm.messages_pb2.Batch()
                num_docs_batch = 0
                batch_token_id = 0
                batch_dict     = defaultdict(int)
                NNZ            = 0
            if num_docs_curr == num_docs_total: # all documents have been found
                break 
            
if __name__ == '__main__':
    batch_path = 'batches_test_1/' 
    
    (doc_set_ru, doc_set_en) = load_doc_sets(csv_path='ru2en.csv')
    print len(doc_set_ru), len(doc_set_en)

    input_ru = '/home/ubuntu/ruwiki-20141203-pages-articles.xml.bz2'
    #input_ru = '/home/ubuntu/ruwiki-20141203-pages-articles1.xml.bz2'
    save_to_batches(input_ru, doc_set_ru, batch_path="ru_batches/", lang='@russian', batch_size = 1000)

    input_en = '/home/ubuntu/enwiki-20141208-pages-articles.xml.bz2'
    #input_en = '/home/ubuntu/enwiki-20141208-pages-articles1.xml-p000000010p000010000.bz2'
    save_to_batches(input_en, doc_set_en, batch_path="en_batches/", lang='@english', batch_size = 1000)
