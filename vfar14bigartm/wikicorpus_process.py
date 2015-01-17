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


def load_title_list(csv_path='ru2en.csv'):
    title_list = list()
    with open(csv_path, 'r') as csv_file:
        for line in csv_file:
            (id_ru, title_ru, id_en, title_en) = line.split('|') 
            title_list.append((unicode(title_ru, 'utf-8'), unicode(title_en, 'utf-8')))
    print 'Doc list done'
    return title_list


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
       
            num_docs_batch += 1
            num_docs_curr  += 1
            if (num_docs_batch == batch_size) or\
               (num_docs_curr == num_docs_total and num_docs_batch <> 0):
                for token in batch_dict: 
                    batch.token.append(unicode(token, 'utf-8'))
                artm.library.Library().SaveBatch(batch, batch_path)
                batch = artm.messages_pb2.Batch()
                num_docs_batch = 0
                batch_token_id = 0
                batch_dict     = defaultdict(int)
                print 'Batch done'
            if num_docs_curr == num_docs_total: # all documents have been found
                break 
            
def merge_batches(title_list, batch_path='.', batch_size=1000):
    def convert_lang_item(batch_id, item_id, lang, dict_lang_to_all):
        item_lang = batch_list[batch_id].item[item_id]
        field = item.field.add()
        field.name = lang
        for token_num in xrange(item_lang.field.token_id):
            token_id = item_lang.field.token_id[token_num]
            token_count = item_lang.field.token_count[token_num]
            field.token_id.append(batch_token_id)
            field.token_count.append(token_count)
            if token_id not in dict_lang_to_all:
                dict_lang_to_all[token_id] = batch_token_id
                batch_token_list.append(batch_list[batch_id].token[token_id], lang)
                batch_token_id += 1
        return dict_lang_to_all
    
    # load batches
    os.chdir(batch_path)
    batch_list = list()
    doc_title_to_batch_id = dict()
    for batch_file in glob.glob("*.batch"):
        batch = artm.library.Library().LoadBatch(batch_file)
        batch_id = len(batch_list)
        for (item_id, item) in enumerate(batch.item):
            print item.title
            doc_title_to_batch_id[item.title] = (batch_id, item_id)
        batch_list.append(batch)   
    
    dict_ru_to_all = dict()
    dict_en_to_all = dict()
    batch_token_list = list()
    batch_token_id   = 0
    
    num_titles_all = len(title_list)
    num_docs_batch = 0
    batch_dict     = defaultdict(int)
    batch_token_id = 0
    batch = artm.messages_pb2.Batch()    
    for (title_ru, title_en) in title_list:
#         print title_ru, title_en
#         if title_ru in doc_title_to_batch_id:
#             print title_ru 
        if title_en in doc_title_to_batch_id:
            print title_en
        if title_ru in doc_title_to_batch_id and title_en in doc_title_to_batch_id:
            print num_docs_batch, title_ru, title_en
            item = batch.item.add()
                        
#             item_ru = batch_list[batch_id_ru].item[item_id_ru]
#             field = item.field.add()
#             field.name = '@russian'
#             for token_num in xrange(item_ru.field.token_id):
#                 token_id = item_ru.field.token_id[token_num]
#                 token_count = item_ru.field.token_count[token_num]
#                 field.token_id.append(dict_all_id)
#                 field.token_count.append(token_count)
#                 dict_ru_to_all[token_id] = dict_all_id
#                 dict_all.append(batch_list[batch_id_ru].token[token_id], '@russian')
#                 dict_all_id += 1
            (batch_id_ru, item_id_ru) = doc_title_to_batch_id[title_ru]
            dict_ru_to_all = convert_lang_item(batch_id_ru, item_id_ru, '@russian', dict_ru_to_all)
            
            (batch_id_en, item_id_en) = doc_title_to_batch_id[title_en]
            dict_en_to_all = convert_lang_item(batch_id_en, item_id_en, '@english', dict_en_to_all)
            
            num_docs_batch += 1
            if (num_docs_batch == batch_size) or\
               ((title_ru, title_en) == title_list[num_titles_all] and num_docs_batch <> 0):
                for (token, lang) in batch_token_list: 
                    batch.token.append(unicode(token, 'utf-8'))
                    batch.class_id.append(lang)
                artm.library.Library().SaveBatch(batch, 'batches2/')
                batch = artm.messages_pb2.Batch()
                num_docs_batch = 0
                batch_token_list = list()
                batch_token_id = 0
                print 'Batch done'


if __name__ == '__main__':
    batch_path = 'batches_test_1/' 
    
    title_list = load_title_list(csv_path='ru2en.csv')
    merge_batches(title_list, batch_path)
#     (doc_set_ru, doc_set_en) = load_doc_sets(csv_path='ru2en.csv')
#     print len(doc_set_ru), len(doc_set_en)
#      
#     input_ru = r'E:\DATA_TEXT\Wiki\ruwiki-20150110-pages-articles.xml.bz2'
#     save_to_batches(input_ru, doc_set_ru, batch_path=batch_path, lang='@russian')
#      
#     input_en = r'E:\DATA_TEXT\Wiki\enwiki-20141208-pages-articles.xml.bz2'
#     save_to_batches(input_en, doc_set_en, batch_path=batch_path, lang='@english')
        
