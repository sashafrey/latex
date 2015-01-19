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


def load_title_list(csv_path='ru2en.csv'):
    title_list = list()
    with open(csv_path, 'r') as csv_file:
        for line in csv_file:
            (id_ru, title_ru, id_en, title_en) = line.split('|') 
            title_list.append((unicode(title_ru, 'utf-8').strip(' \t\n\r'), unicode(title_en, 'utf-8').strip(' \t\n\r')))
    print 'Doc list done'
    return title_list


class BatchConverter:
    def __init__(self):
        return
            
    def convert_lang_item(self, batch_id, item_id, lang, dict_lang_to_all, item):
        item_lang = self.batch_list[batch_id].item[item_id]
        item_lang_field = item_lang.field[0]
        field = item.field.add()
        field.name = lang
        for token_num in xrange(len(item_lang_field.token_id)):
            token_id = item_lang_field.token_id[token_num]
            token_count = item_lang_field.token_count[token_num]
            field.token_id.append(self.batch_token_id)
            field.token_count.append(token_count)
            if token_id not in dict_lang_to_all:
                dict_lang_to_all[token_id] = self.batch_token_id
                self.batch_token_list.append((self.batch_list[batch_id].token[token_id], lang))
                self.batch_token_id += 1
                return dict_lang_to_all

    def merge_batches(self, title_list=None, batch_path='./', batch_size=1000):
        # load batches
        self.batch_list = list()
        doc_title_to_batch_id = dict()
        all_batch_names = glob.glob(batch_path + "*.batch")
        for batch_file in all_batch_names:
            batch = artm.library.Library().LoadBatch(batch_file)
            batch_id = len(self.batch_list)
            for (item_id, item) in enumerate(batch.item):
                doc_title_to_batch_id[item.title] = (batch_id, item_id)
            self.batch_list.append(batch)   
            print str(len(self.batch_list)) + " of " + str(len(all_batch_names)) + " batches done."
#            if (len(self.batch_list) == 40):
#                break

        self.dict_ru_to_all = dict()
        self.dict_en_to_all = dict()
        self.batch_token_list = list()
        self.batch_token_id   = 0

        num_titles_all = len(title_list)
        num_docs_batch = 0
        batch_dict     = defaultdict(int)
        batch = artm.messages_pb2.Batch()    

        ru_found_cnt = 0
        en_found_cnt = 0
        xx_found_cnt = 0

        processed = 0
        for (title_ru, title_en) in title_list:
            processed += 1
            ru_found = (title_ru in doc_title_to_batch_id)
            en_found = (title_en in doc_title_to_batch_id)

            if (ru_found and en_found):
                ru_batch_id = doc_title_to_batch_id[title_ru]
                en_batch_id = doc_title_to_batch_id[title_en]
                if ru_batch_id == en_batch_id:
                  continue;
            
            if (ru_found):
                ru_found_cnt+=1
            if (en_found):
                en_found_cnt+=1
            if (ru_found and en_found):
                xx_found_cnt+=1

            if (ru_found and en_found):
                print num_docs_batch, title_ru, title_en
                item = batch.item.add()

                (batch_id_ru, item_id_ru) = doc_title_to_batch_id[title_ru]
                self.dict_ru_to_all = self.convert_lang_item(batch_id_ru, item_id_ru, '@russian', self.dict_ru_to_all, item)

                (batch_id_en, item_id_en) = doc_title_to_batch_id[title_en]
                self.dict_en_to_all = self.convert_lang_item(batch_id_en, item_id_en, '@english', self.dict_en_to_all, item)

                num_docs_batch += 1

            if (num_docs_batch == batch_size) or (processed == num_titles_all and num_docs_batch <> 0):
                for (token, lang) in self.batch_token_list: 
                    batch.token.append(token)
                    batch.class_id.append(lang)
                artm.library.Library().SaveBatch(batch, 'batches2')
                batch = artm.messages_pb2.Batch()
                num_docs_batch = 0
                self.batch_token_list = list()
                self.batch_token_id = 0
                print 'Batch done'

        print str(ru_found_cnt) + " " + str(en_found_cnt) + " " + str(xx_found_cnt)


if __name__ == '__main__':
    
    title_list = load_title_list(csv_path='ru2en.csv')
    bc = BatchConverter()
    bc.merge_batches(title_list = title_list, batch_path = 'results_copy/', batch_size = 1000)
