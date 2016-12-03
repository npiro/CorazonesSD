# coding: utf-8
#
## Import necessary stuff
import pandas as pd
import os.path
import numpy as np
from joblib import Parallel, delayed
import time
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


processors = 4
max_chunksize=10000
csv_name = 'train_ver2'
##train_csv = os.path.join(dir,'train_ver2.csv')


def main():
    print 'n_lines = ', load_dataset_line_number()
    client_ids = get_client_ids()
    print 'n_clients = ', client_ids.shape

    client_index_array = np.load('../' + csv_name + '__client_indices.npy')
    dataset_line_number = load_dataset_line_number()
    print 'dataset_line_number =',dataset_line_number
    dataset_line_number = load_dataset_line_number()
    
    train_chunks = float(dataset_line_number)/max_chunksize
    
    product_ids = get_product_ids()
    #create_utility_matrix(product_ids)
    utility_matrix = load_utility_matrix()
           
    mean_user_rating = np.nanmean(utility_matrix,axis=0)
#    print product_ids
#    print mean_user_rating
    

    
    # Replace NaNs by zeros
    utility_matrix[np.isnan(utility_matrix)] = 0
    
#    from sklearn.metrics.pairwise import pairwise_distances
#    user_similarity = pairwise_distances(utility_matrix_sparse, metric='cosine',dense_output=False)
#    print user_similarity

    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import sparse
    utility_matrix_sparse = sparse.csr_matrix(utility_matrix[:,:])
    
    #user_similarity = cosine_similarity(utility_matrix_sparse,dense_output=False)
    #np.save(csv_name + '__user_similarity_30000',user_similarity)
    #print user_similarity
    
    #print utility_matrix.shape

    
    item_prediction = predict_memory_based(utility_matrix, type='item')
    

def predict_memory_based(ratings, type='item'):
    # Replace NaNs by zeros
    ratings[np.isnan(ratings)] = 0
    
    ratings_sparse = sparse.csr_matrix(ratings[:,:])

#    from sklearn.metrics.pairwise import pairwise_distances
#    user_similarity = pairwise_distances(utility_matrix_sparse, metric='cosine',dense_output=False)

    if type == 'user':
        #mean_user_rating = ratings.mean(axis=1)
        ##You use np.newaxis so that mean_user_rating has same format as ratings
        #ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        #pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        pass
    elif type == 'item':
        similarity = cosine_similarity(ratings_sparse.T,dense_output=False)
        pred = ratings_sparse.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])[0,:,0]     
    return pred

#pairwise_distances(utility_matrix_sparse.T, metric='cosine',dense_output=False)
#    print item_similarity


def get_train_reader():
    return pd.read_csv(os.path.join('..',csv_name+'.csv'), chunksize=max_chunksize, encoding = 'latin1')


def create_dataset_line_number():
    import shlex
    from subprocess import Popen, PIPE
    cmd = "wc -l " + '../'+csv_name + '.csv'
    process = Popen(shlex.split(cmd), stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait
    train_lines = int(output.split()[0])
    np.save(csv_name + '__line_number',train_lines)
#    return train_lines


def load_dataset_line_number():
    return np.load(csv_name + '__line_number.npy')


def get_client_ids():
    try:
        return np.load('client_ids.npy')
    except ValueError:
        return create_client_ids()
        
def create_client_ids():
    client_ids = np.array([],dtype='int')
    for i, chunk in enumerate(train_reader):
        client_ids = np.union1d(client_ids,chunk.ncodpers.unique())
    np.save('client_ids',client_ids)
    return client_ids

#class train_chunker(object):
#    """
#    """
#    def __init__(self,chunksize):
#        self.chunksize     = chunksize
#        self.csv_path      = os.path.join(dir,'train_ver2.csv')
#        self.client_ids    = np.load('client_ids.npy')
#        self.client_number = self.client_ids.shape[0]
#
#        self.chunks = int(np.ceil(float(self.client_number)/self.chunksize))
#        print(str(self.chunks) + ' chunks')
#        self.reader = pd.read_csv(self.csv_path, chunksize=self.chunksize)


def find_client_index(ncodpers):
    return np.argwhere(client_ids == ncodpers)


def create_dataset_client_index_array():

    train_data_client_indices = np.array([],dtype='int')
    #train_data_client_indices = np.full((client_number),np.NaN,dtype='int')
    j = 0

    # measure process time
    import time
    t0 = time.clock()

    train_reader = get_train_reader()
    for i, chunk in enumerate(train_reader):
        if i%100==0:
            proc_time = time.clock() - t0
            print('chunk ' + str(i) + ' from ' + str(train_chunks) + '. ('\
             + str(proc_time) + ' secs)')

        chunk_size = len(chunk)
        chunk_ncodpers = chunk.ncodpers
        #print chunk_ncodpers[0]
        stdout = Parallel(n_jobs=processors)\
        (delayed(find_client_index)(ncodpers) for ncodpers in chunk_ncodpers)

        chunk_indices = np.full((chunk_size),np.NaN)
        for i, indice in enumerate(stdout):
            chunk_indices[i]= indice

        train_data_client_indices = \
        np.concatenate((train_data_client_indices, chunk_indices))
        #    train_data_client_indices[i*chunksize:(i+1)*chunksize] = chunk_indices

    np.save(csv_name + '__client_indices',train_data_client_indices)


def get_product_ids():
    train_reader = get_train_reader()
    for i, chunk in enumerate(train_reader):
        column_ids = chunk.keys()
    #            print vars(this_chunk).iteritems()
        product_ids = [s for s in column_ids if "ult1" in s]
        break
    return product_ids


def create_utility_matrix(product_ids):
    print sys._getframe().f_code.co_name

    train_reader = get_train_reader()

#    utility_matrix = np.zeros((client_number,len(product_ids)),dtype='int')
    utility_matrix = np.zeros((client_number,len(product_ids)))
    
    t0 = time.clock()
    j = 0
    for i, chunk in enumerate(train_reader):
        if i%1==0:
            proc_time = time.clock() - t0
            print('chunk ' + str(i) + ' from ' + str(train_chunks) + '. (' + str(proc_time) + ' secs)')

        chunk_len = len(chunk)
        chunk_client_indices = client_index_array[j:j+chunk_len]
        j+=chunk_len
        
        for il in range(chunk_len):
            for ip, product_id in enumerate(product_ids):
                utility_matrix[chunk_client_indices[il],ip] += getattr(chunk,product_id)[il]
    
    np.save(csv_name + '__utility_matrix',utility_matrix)
    

def load_utility_matrix():
    return np.load(csv_name + '__utility_matrix.npy')
 

def get_client_matrix():
    train_reader = get_train_reader()

    client_id = client_ids[0]
    client_matrix = pd.DataFrame()
    for i, chunk in enumerate(train_reader):
        client_lines = chunk.ncodpers == client_id
        #print client_lines == True
        client_matrix = client_matrix.append(chunk.ix[client_lines])    
        isinchunk = np.sum(client_lines)
    
    #    if isinchunk != 0.0:
    #        print i, isinchunk,len(client_matrix)
    #        #print chunk.ix[client_lines]
    #        print client_matrix
    #        raw_input()
            #break
    return client_matrix

            
if __name__ == '__main__':
    main()
