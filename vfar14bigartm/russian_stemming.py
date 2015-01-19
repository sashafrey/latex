# -*- coding: utf-8 -*-

from collections import defaultdict
import artm.messages_pb2, artm.library 
import os
import glob
import pymystem3


def stem_russian_batches(batch_path='.', batch_stem_path='./stem/'):
    stemmer = pymystem3.Mystem()
    os.chdir(batch_path)
    for batch_file in glob.glob("*.batch"):
        print batch_file
        batch = artm.library.Library().LoadBatch(batch_file)
        batch_stem = artm.messages_pb2.Batch() 
        # stem tokens
        token_list = list()
        for token in batch.token:
            token_list.append(token)
        text = ' '.join(token_list)
        text_stem = stemmer.lemmatize(text)
        token_stem_list = ''.join(text_stem).strip().split(' ')
        print len(token_stem_list)
        token_id_to_token_stem_id = dict()
        token_stem_to_token_stem_id = dict()
        for (token_id, token_stem) in enumerate(token_stem_list):
#             print token_id, token_stem
            if token_stem not in batch_stem.token:
                token_stem_to_token_stem_id[token_stem] = len(batch_stem.token)
                batch_stem.token.append(token_stem)
            token_id_to_token_stem_id[token_id] = token_stem_to_token_stem_id[token_stem]
        # convert items
        for item in batch.item:
            print item.title
            # add item
            item_stem = batch_stem.item.add()
            item_stem.id = item.id 
            item_stem.title = item.title   
            # add fields
            for field in item.field:
                field_stem_dict = defaultdict(int)
                for token_num in xrange(len(field.token_id)):
                    token_id = field.token_id[token_num]
                    token_stem_id = token_id_to_token_stem_id[token_id]
                    token_count = field.token_count[token_num]
                    field_stem_dict[token_stem_id] += token_count 
    
                field_stem = item_stem.field.add()
                field_stem.name = field.name
                for token_stem_id in field_stem_dict:
                    field_stem.token_id.append(token_stem_id)
                    field_stem.token_count.append(field_stem_dict[token_stem_id])
        # save batch
        artm.library.Library().SaveBatch(batch_stem, batch_stem_path)

if __name__ == '__main__':
    batch_path = 'batches/'
    batch_stem_path='batches_test_1/stem/' 

    stem_russian_batches(batch_path)

