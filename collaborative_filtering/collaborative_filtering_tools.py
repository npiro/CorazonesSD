# coding: utf-8
import pandas as pd
import os.path
import numpy as np
from joblib import Parallel, delayed
import time
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import shlex
from subprocess import Popen, PIPE

processors = 4
max_chunksize=10000
data_dir = '..'


def main():
    pass


def get_data_dict(csv_name):
    data = dict()
    data['frame']      = get_data_frame(csv_name)
    data['n_lines']    = get_n_lines(csv_name)
    data['client_ids'] = get_client_ids(csv_name)
    data['n_clients']  = data['client_ids'].shape[0]
    data['n_chunks']   = float(data['n_lines'])/max_chunksize
    data['client_indices'] = get_client_indices(csv_name)
    return data
    

def get_data_frame(csv_name):
    return pd.read_csv(os.path.join(data_dir,csv_name+'.csv')
    , chunksize=max_chunksize, encoding = 'latin1')
    
    
def get_n_lines(csv_name):
    n_lines_file = os.path.join(data_dir,csv_name+'_n_lines')
    try:
        return int(np.load(n_lines_file+'.npy'))
    except IOError:
        data_file = os.path.join(data_dir,csv_name+'.csv')
        cmd = "wc -l " + data_file
        process = Popen(shlex.split(cmd), stdout=PIPE)
        (output, err) = process.communicate()
        #exit_code = process.wait
        data_n_lines = int(output.split()[0])
        np.save(n_lines_file,data_n_lines)
        return data_n_lines
        

def get_product_ids(csv_name):
    data_frame = get_data_frame(csv_name)
    for i, chunk in enumerate(data_frame):
        column_ids = chunk.keys()
        product_ids = [s for s in column_ids if "ind" in s and "ult1" in s ]
        break
    return product_ids


def get_client_ids(csv_name):
    client_ids_file = os.path.join(data_dir,csv_name+'_client_ids.npy')
    try:
        return np.load(client_ids_file)
    except IOError:
        data_frame = get_data_frame(csv_name)
        client_ids = np.array([],dtype='int')
        for i, chunk in enumerate(data_frame):
            client_ids = np.union1d(client_ids,chunk.ncodpers.unique())
        np.save(client_ids_file,client_ids)
        return client_ids
        

client_ids = get_client_ids('train_ver2')
def find_client_index(ncodpers):
    return np.argwhere(client_ids == ncodpers)


def get_client_indices(csv_name):
    client_indices_file = os.path.join(data_dir,csv_name+'_client_indices.npy')
    try:
        return np.load(client_indices_file)
    except IOError:

        data_client_indices = np.array([],dtype='int')
        t0 = time.clock()
    
        data  = get_data_dict(csv_name)
    
        for i, chunk in enumerate(data['frame']):
            if i % int(data['n_chunks']/10) == 0:
                proc_time = time.clock() - t0
                print('chunk ' + str(i) + ' from ' + str(data['n_chunks'])\
                + '. (' + str(proc_time) + ' secs)')
    
            chunk_size = len(chunk)
            chunk_ncodpers = chunk.ncodpers
            #print chunk_ncodpers[0]
            stdout = Parallel(n_jobs=processors)\
            (delayed(find_client_index)(ncodpers) for ncodpers in chunk_ncodpers)
    
            chunk_indices = np.full((chunk_size),np.NaN)
            for i, indice in enumerate(stdout):
                chunk_indices[i]= indice
    
            data_client_indices = \
            np.concatenate((data_client_indices, chunk_indices))
            #    train_data_client_indices[i*chunksize:(i+1)*chunksize] = chunk_indices
    
        np.save(client_indices_file,data_client_indices)
        return data_client_indices


def get_utility_matrix(csv_name,product_ids,client_indices):
    utility_matrix_file = os.path.join(data_dir,csv_name+'_utility_matrix.npy')
    try:
        return np.load(utility_matrix_file)
    except IOError:

        data  = get_data_dict(csv_name)
        utility_matrix = np.zeros((client_ids.shape[0],len(product_ids)))
        
        t0 = time.clock()
        j = 0
        for i, chunk in enumerate(data['frame']):
            if i%1==0:
                proc_time = time.clock() - t0
                print('chunk ' + str(i) + ' from ' + str(data['n_chunks']) \
                + '. (' + str(proc_time) + ' secs)')
    
            chunk_len = len(chunk)
            chunk_client_indices = client_indices[j:j+chunk_len]
            j+=chunk_len
            
            for il in range(chunk_len):
                for ip, product_id in enumerate(product_ids):
                    utility_matrix[chunk_client_indices[il],ip] \
                    += getattr(chunk,product_id)[il]
        
        np.save(utility_matrix_file,utility_matrix)
        return utility_matrix



def predict_memory_based(ratings, type='item'):

    # Replace NaNs by zeros
    ratings[np.isnan(ratings)] = 0
    
    if type == 'user':
        # user cosine similarity calculation requires a lot of memory
        # could not be computed in my laptop 
        ratings_sparse = sparse.csr_matrix(ratings[:,:])
        similarity = cosine_similarity(ratings_sparse,dense_output=False)
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

    elif type == 'item':
        ratings_sparse = sparse.csr_matrix(ratings[:,:])
        similarity = cosine_similarity(ratings_sparse.T,dense_output=False)
        # or
        #from sklearn.metrics.pairwise import pairwise_distances
        #similarity = pairwise_distances(ratings_matrix.T, metric='cosine')

        # sklearn.metrics.pairwise.pairwise_distances apparently does not 
        # support sparse matrices

        pred = ratings_sparse.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])[0,:,0]
        
    return pred


#    print sys._getframe().f_code.co_name

#def get_train_reader():
#    return pd.read_csv(os.path.join('..',csv_name+'.csv'), chunksize=max_chunksize, encoding = 'latin1')


#def create_dataset_line_number():
#    import shlex
#    from subprocess import Popen, PIPE
#    cmd = "wc -l " + '../'+csv_name + '.csv'
#    process = Popen(shlex.split(cmd), stdout=PIPE)
#    (output, err) = process.communicate()
#    exit_code = process.wait
#    train_lines = int(output.split()[0])
#    np.save(csv_name + '__line_number',train_lines)
##    return train_lines
#
#
#def load_dataset_line_number():
#    return np.load(csv_name + '__line_number.npy')
#


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
