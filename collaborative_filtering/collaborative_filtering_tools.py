# coding: utf-8
import pandas as pd
#import os.path
from os.path import join
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
    #data['client_indices'] = get_client_indices(csv_name)
    return data
    

def get_data_frame(csv_name):
    return pd.read_csv(join(data_dir,csv_name+'.csv')
    , chunksize=max_chunksize, encoding = 'latin1')


def run_in_terminal(cmd):
    print cmd
    output=dict()
    process = Popen(shlex.split(cmd), stdout=PIPE)
    (output['stdout'],output['stderr']) = process.communicate()
    output['exit_code'] = process.wait
    return output


    
def progress_message(i,t0,n_chunks):
    if i % int(n_chunks/10) == 0:
        proc_time = float(time.clock() - t0)/60
        print('chunk ' + str(i) + ' from ' + str(n_chunks) \
        + '. (' + str(proc_time) + ' minutes)')

    
def get_n_lines(csv_name):
    n_lines_file = join(data_dir,csv_name+'_n_lines')
    try:
        return int(np.load(n_lines_file+'.npy'))
    except IOError:
        print 'creating '+ n_lines_file
        data_file = join(data_dir,csv_name+'.csv')
        output = run_in_terminal("wc -l " + data_file)        
        data_n_lines = int(output['stdout'].split()[0])
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
    client_ids_file = join(data_dir,csv_name+'_client_ids.npy')
    try:
        return np.load(client_ids_file)
    except IOError:
        print 'creating '+ client_ids_file
        data_frame = get_data_frame(csv_name)
        client_ids = np.array([],dtype='int')
        for i, chunk in enumerate(data_frame):
            client_ids = np.union1d(client_ids,chunk.ncodpers.unique())
        np.save(client_ids_file,client_ids)
        return client_ids


def get_unique_dates(csv_name):
    data  = get_data_dict(csv_name)
    unique_dates = np.array([],dtype='a10')

    t0 = time.clock()
    for i, chunk in enumerate(data['frame']):
        progress_message(i,t0,data['n_chunks'])
        unique_dates = np.union1d(unique_dates,chunk.fecha_dato.unique())

    return unique_dates
        

def find_first_date_appeareance_line(csv_name,date_to_find):
    data  = get_data_dict(csv_name)

    line=1
    t0 = time.clock()
    for i, chunk in enumerate(data['frame']):
        progress_message(i,t0,data['n_chunks'])
        if date_to_find in chunk.fecha_dato.unique():
            for j,date in enumerate(chunk.fecha_dato):
                if date == date_to_find:
                    line += j
                    return line

        line += len(chunk)


def find_client_index((client_ids,ncodpers)):
    return np.argwhere(client_ids == ncodpers)


def get_client_indices(csv_name):
    client_indices_file = join(data_dir,csv_name+'_client_indices.npy')
    try:
        return np.load(client_indices_file)
    except IOError:
        print 'creating '+ client_indices_file
        data_client_indices = np.array([],dtype='int')
        t0 = time.clock()
    
        data  = get_data_dict(csv_name)
        client_ids = get_client_ids(csv_name)
    
        for i, chunk in enumerate(data['frame']):
            progress_message(i,t0,data['n_chunks'])
            chunk_size = len(chunk)
            find_client_index_input_list = list()
            for ncodpers in chunk.ncodpers:
                find_client_index_input_list.append((client_ids,ncodpers))
            
            #print chunk_ncodpers[0]
            stdout = Parallel(n_jobs=processors)\
            (delayed(find_client_index)((client_ids,ncodpers))\
            for (client_ids,ncodpers) in find_client_index_input_list)
    
            chunk_indices = np.full((chunk_size),np.NaN)
            for i, indice in enumerate(stdout):
                chunk_indices[i]= indice
    
            data_client_indices = \
            np.concatenate((data_client_indices, chunk_indices))
            #    train_data_client_indices[i*chunksize:(i+1)*chunksize] = chunk_indices
    
        np.save(client_indices_file,data_client_indices)
        return data_client_indices


def get_utility_matrix(csv_name):
    utility_matrix_file = join(data_dir,csv_name+'_utility_matrix.npy')
    client_n_months_file = join(data_dir,csv_name+'_client_n_months.npy')
    try:
        utility_matrix = np.load(utility_matrix_file)
        client_n_months= np.load(client_n_months_file)

    except IOError:
        print 'creating '+ utility_matrix_file

        data           = get_data_dict(csv_name)
        product_ids    = get_product_ids(csv_name)
        client_indices = get_client_indices(csv_name)
        utility_matrix  = np.zeros((data['client_ids'].shape[0],len(product_ids)))
        client_n_months = np.zeros((data['client_ids'].shape[0],1))
        
        t0 = time.clock()
        j = 0
        for i, chunk in enumerate(data['frame']):
            progress_message(i,t0,data['n_chunks'])
    
            chunk_len = len(chunk)
            chunk_client_indices = client_indices[j:j+chunk_len]
            j+=chunk_len
            
            for il in range(chunk_len):
                for ip, product_id in enumerate(product_ids):
                    client_n_months[chunk_client_indices[il]] += 1
                    utility_matrix[chunk_client_indices[il],ip] \
                    += getattr(chunk,product_id)[il]

        # normalize utility_matrix entries
        for j in range(utility_matrix.shape[0]):
            utility_matrix[j,:] = utility_matrix[j,:] / client_n_months[j]

        np.save(utility_matrix_file,utility_matrix)
        np.save(client_n_months_file,client_n_months)
    
    return utility_matrix, client_n_months


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


#-------------------------------------------------------------------------------
# the notebook split_train_dataset.ipynb is equivalent to this function
# and works well, here head and tail programs always give errors   
#-------------------------------------------------------------------------------
#def split_train_dataset():
    #"""
    #as test_ver2.csv dataset did not have product columns a new train-test
    #dataset pair (train_ver2_month1-16.csv,train_ver2_month17.csv) was created
    #"""
    #month17_first_line = find_first_date_appeareance_line('train_ver2','2016-05-28')
    ##print month17_first_line  
    #data  = get_data_dict('train_ver2')
    #last_month_lines = data['n_lines']- month17_first_line
    #f1 = join(data_dir,'train_ver2.csv')
    #f2 = join(data_dir,'train_ver2_month1-16.csv')
    #f3 = join(data_dir,'train_ver2_month17.csv')
    #run_in_terminal('head -n ' + str(month17_first_line)+' "'+ f1 +'"> "'+ f2+'"')
    #run_in_terminal('head -n 1'+                         ' "'+ f1 +'"> "'+ f3+'"')
    #run_in_terminal('tail -n ' + str(last_month_lines)  +' "'+ f1 +'">>"'+ f3+'"')
#-------------------------------------------------------------------------------

#    print sys._getframe().f_code.co_name

#def get_client_matrix():
#    train_reader = get_train_reader()
#
#    client_id = client_ids[0]
#    client_matrix = pd.DataFrame()
#    for i, chunk in enumerate(train_reader):
#        client_lines = chunk.ncodpers == client_id
#        #print client_lines == True
#        client_matrix = client_matrix.append(chunk.ix[client_lines])    
#        isinchunk = np.sum(client_lines)
#    
#    #    if isinchunk != 0.0:
#    #        print i, isinchunk,len(client_matrix)
#    #        #print chunk.ix[client_lines]
#    #        print client_matrix
#    #        raw_input()
#            #break
#    return client_matrix

            
if __name__ == '__main__':
    main()
