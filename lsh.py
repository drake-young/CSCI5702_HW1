# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla
# Modified by Evan Stene for CSCI 5702/7702
# Modified by Drake Young for submission of Assignment 1 in CSCI5702

import unittest
import numpy                 as np
import pyspark.sql.functions as F
from PIL         import Image
from pyspark     import SparkContext
from pyspark.sql import SQLContext
from timeit      import default_timer

class LSH:

    def __init__( self , A , k , L ):
        """
        Initializes the LSH object
        A - dataframe to be searched
        k - number of thresholds in each function
        L - number of functions
        """
        label  =  '[LSH]'
        
        print('%8s Establishing Spark Context...' %  label )
        self.sc          =  SparkContext( )
        self.sqlContext  =  SQLContext( self.sc )
        self.k           =  k
        self.L           =  L
        
        print( '%8s Constructing Dataframe...' % label )
        self.A  =  self.load_data( A )
        
        print( '%8s Creating Hash Functions...' % label )
        self.functions  =  self.create_functions( )
        
        print( '%8s Hashing Dataframe...' % label )
        self.hashed_A  =  self.hash_data( A )
        
        
    def l1( u , v ):
        """
        Finds the L1 distance between two vectors
        u and v are 1-dimensional Row objects
        """
        # Taxicab Geometry: distance is the sum of absolute difference of 
        # Cartesian coordinates (sum the abs difference between u[i] and v[i]
        # for every i in u and v)
        # Data is Uniform -- All patches are of length 400
        w  =  0
        for i in range( len( v ) ):
            w  +=  abs( float( u[i] ) - float( v[i] ) )
        return w


    def load_data( self , filename ):
        """
        Loads the data into a spark DataFrame, where each row corresponds to
        an image patch -- this step is sort of slow.
        Each row in the data is an image, and there are 400 columns.
        """
        lines    =  self.sc.textFile( filename ) # Construct the RDD from the file
        patches  =  lines.map( lambda l : l.split( ',' ) ) # Parse the file by commas
        rdd      =  patches.zipWithIndex( ) # Add an index column for easy lookup later
        df       =  self.sqlContext.createDataFrame( rdd ) # Convert RDD to Dataframe
        df       =  df.withColumnRenamed( '_2' , 'index' ).withColumnRenamed( '_1' , 'data' ) # Rename cols
        return df
        
    
    def create_function( self , dimensions , thresholds ):
        """
        Creates a hash function from a list of dimensions and thresholds.
        """
        def f( v ):
            # convert into an array of bits
            # 1 -- the vector's value for a specific dimension is greater than the threshold
            # 0 -- less than threshold
            # generate for all dimensions
            # convert array into a single string to be returned
            return ''.join( [ '1' if float( v[ dimensions[i] ] ) >= thresholds[i] 
                                  else '0' 
                                  for i in range( len( dimensions ) ) ] )
        return f


    def create_functions( self , num_dimensions=400 , min_threshold=0 , max_threshold=255 ):
        """
        Creates the LSH functions (functions that compute L K-bit hash keys).
        Each function selects k dimensions (i.e. column indices of the image matrix)
        at random, and then chooses a random threshold for each dimension, between 0 and
        255.  For any image, if its value on a given dimension is greater than or equal to
        the randomly chosen threshold, we set that bit to 1.  Each hash function returns
        a length-k bit string of the form "0101010001101001...", and the L hash functions 
        will produce L such bit strings for each image.
        """
        functions  =  [ ]
        for i in range( self.L ):
            dimensions  =  np.random.randint(
                                    low   =  0, 
                                    high  =  num_dimensions,
                                    size  =  self.k )
            thresholds  =  np.random.randint(
                                    low   =  min_threshold, 
                                    high  =  max_threshold + 1, 
                                    size  =  self.k )

            functions.append( self.create_function( dimensions , thresholds ) )
        return functions


    def hash_vector( self , v ):
        """
        Hashes an individual vector (i.e. image).  This produces an array with L
        entries, where each entry is a string of k bits.
        """
        # you will need to use self.functions for this method
        # hash the vector v on all functions f in self.functions
        # compile these hashed vectors into an array to be returned
        return [ f( v ) for f in self.functions ]


    def hash_data( self , filename ):
        """
        Hashes the data in A, where each row is a datapoint, using the L
        functions in 'self.functions'
        """
        # you will need to use self.A for this method
        # map each hashed vector v from the self.hash_vector to A
        df_hash  =  self.sqlContext.createDataFrame( [    [ v['index'] , self.hash_vector( v['data'] ) ] 
                                                          for v in self.A.toLocalIterator( ) ] )
        df_hash  =  df_hash.withColumnRenamed( '_1' , 'index' ).withColumnRenamed( '_2' , 'hash' )
        return df_hash


    def get_candidates( self , hashed_point , query_index ):
        """
        Retrieve all of the points that hash to one of the same buckets 
        as the query point.  Do not do any random sampling (unlike what the first
        part of this problem prescribes).
        Don't retrieve a point if it is the same point as the query point.
        """
        # you will need to use self.hashed_A for this method
        # use a python generator to generate all indices i where 0<=index<lenght of hashed vector
        # exclude the query index, 
        # and only include i if any row of hashed_A matches the hashed_point
        # convert this generated list into a numpy array upon return
        candidates  =  [ ]
        for row in self.hashed_A.toLocalIterator( ):
            for i in hashed_point:
                if i in row['hash']:
                    if row['index'] != query_index:
                        candidates.append( row['index'] )
                    break
        return self.A.filter( self.A['index'].isin( candidates ) )


    def lsh_search( self , query_index , num_neighbors=10 ):
        """
        Run the entire LSH algorithm
        """
        # 1. Figure out the hash at the query index
        # 2. Get the candidates for nearest neighbors
        # 3. Determine distance of candidates
        # 4. order candidates to closeness
        # 5. extract the appropriate number of desired neighbors
        # 6. Only return the desired number of best candidate rows
        label = '[LSH]'
        
        # 1. Figure out the hash at the query index
        print( '%8s Extracting Query Hash...' % label )
        hashed_points  =  self.hashed_A.filter( F.col( 'index' ) == query_index ).take( 1 )[0]['hash']
        
        # 2. Get the candidates for nearest neighbors
        print('%8s Determining Candidates...' % label )
        candidates  =  self.get_candidates( hashed_points , query_index )
        
        # 3. Determine the distance of candidates
        # for every candidate, do the distance ly between A at that index, and A at the query index
        print('%8s Extracting Query Vector...' % label )
        query_vect_df  =  self.A.filter( F.col('index') == query_index )
        query_vect     =  query_vect_df.take( 1 )[0]['data']
        
        print( '%8s Calculating Candidate Distances to Query...' % label )
        distances  =  [ [ row['index'] , LSH.l1( query_vect , row.data ) ] 
                          for row in candidates.toLocalIterator( ) ]
        distances  =  self.sqlContext.createDataFrame( distances )
        distances  =  distances.withColumnRenamed( '_1' , 'index' ).withColumnRenamed( '_2' , 'distance' )
        
        # 4. Order Candidates by Closeness
        print( '%8s Organizing Candidates by Distance...' % label )
        candidates  =  candidates.join( distances , ['index'] )
        candidates  =  candidates.orderBy( F.col('distance') )
        
        # 5. Extract appropriate number of candidates
        print( '%8s Taking %i Nearest Candidates...' % ( label , num_neighbors ) )
        n_candidates  =  candidates.limit( num_neighbors )
        
        # Extract total distance for error calculation
        total_dist = sum( [ row.distance for row in n_candidates.toLocalIterator( ) ] )

        # Return the results        
        return query_vect_df, n_candidates, total_dist
    
    
# Plots images at the specified rows and saves them each to files.
def plot(neighbor_df, base_filename, mark_closest=False, sc=None):
    label = '[PLOT]'
    
    # Establish Spark Context if None Provided
    if sc == None:
        sc  =  SparkContext( )
    
    # Iterate through all rows in dataframe
    for row in neighbor_df.toLocalIterator( ):
        print( '%8s Plotting Patch %i' % ( label , row['index'] ) )
        
        # Extract patch from row, reformat into 20x20, and convert to Image object
        row_data  =  np.array( row['data'] ).astype( np.float )
        patch     =  np.reshape( row_data , [ 20 , 20 ] )
        im        =  Image.fromarray( patch )
        
        # Ensure image mode is RGB before PNG creation
        if im.mode != 'RGB':
            im  =  im.convert( 'RGB' )
            
        # If flagged to mark the closest, add "-closest" into the filename
        if mark_closest:
            outfilename   =  base_filename + '-' + str( row['index'] ) + '-closest.png'
            mark_closest  =  False
        else:
            outfilename   =  base_filename + '-' + str( row['index'] ) + '.png'
        
        # Save the image
        im.save( outfilename )
    return


# Finds the nearest neighbors to a given vector, using linear search.
def linear_search( A , query_index , num_neighbors=10 , sc=None , sqlContext=None ):
    # Establish Spark and SQL Contexts (to package output for return)
    if sc == None:
        sc  =  SparkContext( )
    if sqlContext == None:
        sqlContext  =  SQLContext( sc )
        
    # Extract Query Vector
    query_vect  =  A.filter( F.col('index') == query_index ).take( 1 )[0]['data']
    
    # Compute Distance to Each
    distances  =  [ ]
    for row in A.toLocalIterator( ):
        # don't include the query
        if row['index'] == query_index:
            continue

        # Extract the relevant data from the row
        index     =  row['index']
        distance  =  LSH.l1( query_vect , row.data )

        # This elimination step only works once we found n candidates
        if len( distances ) >= num_neighbors:
            # Don't bother adding if it's farther than all "current best" neighbors
            if distance > distances[-1][1]:
                continue

        # Insert in sorted order to optimize
        idx = 0
        for i in range( len( distances ) ):
            if distances[i][1] > distance:
                idx  =  i
                break
        distances  =  distances[ : idx] + [ [ index , distance ] ] + distances[ idx : num_neighbors-1 ]
        
    # Convert distances to dataframe
    distances  =  sqlContext.createDataFrame( distances )
    distances  =  distances.withColumnRenamed( '_1' , 'index' ).withColumnRenamed( '_2' , 'distance' )
    distances  =  distances.join( A , ['index'] ).orderBy( ['distance'] )
    total_dist = sum( [ row.distance for row in distances.toLocalIterator( ) ] )
    return distances, total_dist

"""
# Write a function that computes the error measure
def lsh_error(lsh_neighbors, linear_neighbors):
    '''
    From the Assignment PDF:
        error = (1/10) * sum_j-10( [ sum_i-3( d( xij, zj ) ) ]   /   [ sum_i-3( d( xij* , zj ) )] )
        
        { zj | 1 <= j <= 10 } = set of image patches considered (all patches)
            for the case of this assignment, z1 is the closest neighbor to the query,
            z2 is the second closest neighbor to the query, etc
            for this assignment, z should be the image patches at
                {100,200,300,400,500,600,700,800,900,1000}
        {xij} is the set of APPROXIMATE nearest neighbors to the query (LSH search)
        {xij*} is the set of the 3 TRUE nearest neighbors to the query (results from linear)
        
        so error is 1/10 * sum
    '''
    error  =  0
    for i in range( len( lsh_neighbors ) ): # assume same number of searches run for both
        numerator    =  sum( [ row.distance for row in lsh_neighbors[i].toLocalIterator( ) ] )
        denominator  =  sum( [ row.distance for row in linear_neighbors[i].toLocalIterator( ) ] )
        error += numerator / denominator
    return error / 10
"""
def lsh_error( lsh_dist , linear_dist ):
    '''
    From the Assignment PDF:
        error = (1/10) * sum_j-10( [ sum_i-3( d( xij, zj ) ) ]   /   [ sum_i-3( d( xij* , zj ) )] )
        
        { zj | 1 <= j <= 10 } = set of image patches considered (all patches)
            for the case of this assignment, z1 is the closest neighbor to the query,
            z2 is the second closest neighbor to the query, etc
            for this assignment, z should be the image patches at
                {100,200,300,400,500,600,700,800,900,1000}
        {xij} is the set of APPROXIMATE nearest neighbors to the query (LSH search)
        {xij*} is the set of the 3 TRUE nearest neighbors to the query (results from linear)
        
        so error is 1/10 * sum
    '''
    error  =  0
    for i in range( len( lsh_neighbors ) ): # assume same number of searches run for both
        numerator    =  lsh_dist[i]
        denominator  =  linear_dist[i]
        error += numerator / denominator
    return error / 10

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 16])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)

if __name__ == '__main__':
#    unittest.main() ### TODO: Uncomment this to run tests
    """
    Your code here
    """
    label  =  '[MAIN]'
    
    print( '%s Creating LSH Object' % label )
    lsh               =  LSH( A='patches.csv' , k=16 , L=10 )
    query_indices     =  [ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ]
    lsh_neighbors     =  [ ]
    lsh_times         =  [ ]
    linear_neighbors  =  [ ]
    linear_times      =  [ ]
    queries           =  [ ]
    lsh_distances     =  [ ]
    linear_distances  =  [ ]
    
    # Need to run for all query indices to achieve an accurate error measure
    for query_index in query_indices:
        print( '%s LSH Search on Index %i' % ( label , query_index ) )
        start  =  default_timer( )
        query , lsh_neighbor, lsh_dist  =  lsh.lsh_search( query_index , num_neighbors=3 )
        end    =  default_timer( )
        lsh_times.append( end - start )
    
        print( '%s Linear Search on Index %i' % ( label , query_index ) )
        start  =  default_timer( )
        linear_neighbor, linear_dist  =  linear_search( lsh.A , query_index , num_neighbors=3 , sc=lsh.sc , sqlContext=lsh.sqlContext )
        end    =  default_timer( )
        linear_times.append( end - start )
        
        lsh_neighbors.append( lsh_neighbor )
        linear_neighbors.append( linear_neighbor )
        queries.append( query )
        lsh_distances.append( lsh_dist )
        linear_distances.append( linear_dist )
    
    print( '%s Computing Error' % label )
    error  =  lsh_error( lsh_distances , linear_distances )
    
    ''' UNCOMMENT THIS BLOCK IF YOU WISH TO GENERATE IMAGES FROM SEARCH RESULTS
    print( '%s Plotting Query Patch' % label )
    plot( queries[0] , 'output/query' , sc=lsh.sc )
    
    print( '%s Plotting LSH Result Patches' % label )
    plot( lsh_neighbors[0] , 'output/lsh-match' , mark_closest=True , sc=lsh.sc )

    print( '%s Plotting Linear Search Result Patches' % label )
    plot( linear_neighbors[0] , 'output/linear-match' , mark_closest=True , sc=lsh.sc )
    '''
    
    print( '%s Stopping Spark Context' % label )
    lsh.sc.stop( )
    
    print( '\n' )
    print('%s Computation Results' % label )
    print( '%s Average LSH Time: %.3f Seconds' % ( label , sum( lsh_times ) / len( lsh_times ) ) )
    print( '%s Average Linear Search Time: %.3f Seconds' % ( label , sum( linear_times ) / len( linear_times ) ) )
    print( '%s Error: %f' % ( label , error ) )
    
    print( '%s Done' % label )
