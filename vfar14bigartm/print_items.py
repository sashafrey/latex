import glob, artm.messages_pb2, artm.library, os

def print_items(batch):
    for item in batch.item[1:100]:
        print item.title + " : ",
        for field in item.field:
            for (token_id, token_count) in zip(field.token_id, field.token_count):
                print batch.token[token_id] + "(" + str(token_count) + "),",
        print "\n",

#batch = artm.library.Library().LoadBatch("C:\\datasets\\merged_batches_cut.tar\\merged_batches_cut\\001d4cc8-885a-471c-a0f2-528e2021a795.batch")
#batch = artm.library.Library().LoadBatch("D:\\datasets\\multilang_wiki\\en_batches\\0a7046b8-dfd4-4f75-8cdb-21226ef048ff.batch")
#batch = artm.library.Library().LoadBatch("C:\\datasets\\merged_batches_cut_test\\001d4cc8-885a-471c-a0f2-528e2021a795.batch")
batch = artm.library.Library().LoadBatch("C:\\Users\\Administrator\\Documents\\GitHub\\latex\\vfar14bigartm\\en_batches\\6b41c0f3-d8aa-4838-afd6-1fb76252a395.batch")
print_items(batch)
