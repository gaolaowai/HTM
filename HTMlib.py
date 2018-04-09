# -*- coding: utf-8 -*-


'''
First, there needs to come an encoder, which translate data into a scalar array.
the algorithm used for encoding needs to be dterministic, always producing the same result with the same input.

input data ---> encoders ---> spatial pooler ---> temporal memory

Some notes:
cla classifier
anomaly scores/likelihood
anomaly classifier
knn classifier
svm classifier
reconstruction


'''
import numpy as np
import random


# Encoder
# Get min-max of possible data input for encoder type
class encoder():
    def __init__(self, bucketwidth, maxval, minn=0):
        self.minn = minn
        
        # 'wing' will be used to 
        self.wing = (bucketwidth - 1 ) / 2
        
        # This is the capping value on our SDR
        self.maxval = maxval
        
        # Find possible ranges for scalar and store it.
        self.size = maxval - minn
        
        # Create a boolean array, which explicitly uses 1 byte per value.
        # This is done with the intention of keeping memory size down when
        # possible.
        self.arraytemplate = np.zeros((self.size), dtype=np.bool_)
        self.encodedarray = np.zeros((self.size), dtype=np.bool_)
        
    def encode(self, value):       
        # Make a local copy of the arraytemplate... if we have memory on a system,
        # this is cheaper CPU-wise than generating one each turn.     
        encoded_local_copy = np.copy(self.arraytemplate)
        
        # For a given input value, find its index range base on "value - wing"
        # and "value + wing". Add 1 to index limit, as python treats the second
        # index value as wall, rather than including that value.
        # Resulting array is our SDR, which we will then return.
        #
        encoded_local_copy[value - self.wing: value + self.wing + 1] = True
        return encoded_local_copy
    
    def decode(self, column_connections):
        '''
        TODO translate a vector back into a scalar value... used for predictions?
        '''
        length = len(column_connections)
        values = float()
        for i in column_connections:
            values += float(i)
        # Get a mean value?
        result = values / length
        return result
                
    # Can't see where we actually need this, but including it here as an example
    # of how to access the object properties using a method, if needed.
    def getSize(self):
        return self.size
        
        
        
class columnCreate():

    #def createColumn(connpercent, encodersize, randomarray):
    def __init__(self, encodersize, randomarray):    
        '''
        randomly, based on inputspacesize create connections to random indices
        '''
        # Randomize the connections to the input space
        np.random.shuffle(randomarray)
        # Get index of 1 values as the connections list.
        self.connections = np.where(randomarray == 1)[0]
        # For each index, set a random weight
        self.connectionweights = {} # Dictionary (other languages refer to it as hashmap)
        for keys in self.connections:
            self.connectionweights[keys] = random.uniform(0.0, 1.0)
        
	def checkInputSpace(self, encodedarray):
        '''
        "encodedarray" value should have been retrieved using "encoder.encode()" class-method.
        Compare connections index with encoder's input space to find matches.
        Adjust connection weights based on results.
        return a value representing "activated connections score"
        '''
        # Change update increment value semi-randomly for each column
        increment = random.uniform(0.01, 0.3)
        
        # Local variable to store number of matching connections for this column for this turn.
        connection_counter = 0
        
        # Check each connection to see if corresponding index in input space is activated.
        # There are better (faster) ways to do this, but this method is more imperative, allowing 
        # students of HTM to better see the logic of what is happening.
        for i in self.connections:
          if encodedarray[i] == True:
            # +1 for the connection counter
            connection_counter += 1
            # Update the weight value
            self.connectionweights[i] += incrememnt
		
        

class SpatialPooler():

    '''
    # pooler object
    # supposedly needs ~2% density

    Elements:
    connections list
    connections dictionary (track connection weights)
    connection match counter
    win-lose state
    column layer tracker
    repeat for number of columns in spatial pool
    '''
    def __init__(self, poolersize, connpercent, encoder):
        '''
        To create a pool of 100 columns with 50% connection rate to input space, use:
        pool = SpatialPooler(100, 50, encoder)  <--- pass in the encoder
        Pool stores a list of columns in the list pool.columns
        Iterate over the list to execute methods on each item.
        '''
        self.connected_encoder=encoder
        # This will generate a template array, used for column creation.
        # The resulting array is then randomized each time a new column is created,
        # thus yielding our randomized column connections.
        self.encodersize = len(encoder)
        # This sets overall size of connection space.
        N = encodersize 
        K = connpercent # K zeros, N-K ones
        self.arr = np.array([0] * K + [1] * (N-K))
        
        # Create columns, stored to this list, based on the value of "poolersize" variable
        self.columns = [columnCreate(encodersize, self.arr) for i in range(poolersize)]
        

    def getValue(self, activation_threshold, encoder):
        # check each column against the input space        
        # Get values for each column  
        for column in self.columns:
            #value, counts = np.unique(column, return_counts=True)
            #column.connection_match_ctr = dict(zip(value, counts))[1]
            idx = np.where(column == 1)[0]
            for value in idx:
                column.connectionweights[value] += 0.09
            
