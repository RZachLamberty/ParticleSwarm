########################################################################
#                                                                      #
#   ParticleSwarm.py                                                   #
#   11/01/2013                                                         #
#                                                                      #
#   A general wrapper for a particle swarm algorithm                   #
#                                                                      #
########################################################################

import os
import csv
import scipy


#-----------------------#
#   Module Constants    #
#-----------------------#


#-----------------------#
#   Swarm class         #
#-----------------------#

class ParticleSwarm():
    """
    A general wrapper for particle swarm algorithms.  To initialize,
    include:
    
        *** MANDATORY ***
        
        rangeSet    :   N pairs of numbers representing the range min
                        and max of the N parameter spaces in which our 
                        particles are swarming (dimensions N x 2)
        
        particleNum :   The number or particles we are simulating
        
        fitnessFunc :   A R^n ---> R function mapping the parameter
                        space
        
        PSVals      :   The parameters of the simulations:
                            omega
                            psi_d
                            psi_g
        
        *** OPTIONAL ***
        
        maxGenNum   :   The generation at which our routine will stop
        
        saveVals    :   Boolean indicating whether or not to save values
                        from the simulation
        
        saveFile    :   A file name to save the data to (csv format)
        
    """
    
    def __init__( self, rangeSet, particleNum, fitnessFunc, PSVals, maxGenNum = None, saveVals = False, saveFile = None ):
        """
        See class doc
        """
        self.N        = len( rangeSet )
        self.rangeSet = scipy.array( rangeSet )
        self.particleNumber     = particleNum
        self.fitFunc  = fitnessFunc
        
        if len( PSVals ) != 3:
            raise ValueError( "PSVals = {}\nPlease provide *three* parameters".format( PSVals ) )
        else:
            self.omega = PSVals[ 0 ]
            self.phiD  = PSVals[ 1 ]
            self.phiG  = PSVals[ 2 ]
        
        if maxGenNum:
            self.maxGenNum = maxGenNum
        
        if saveVals:
            self.saveVals = saveVals
            if not saveFile:
                raise ValueError( "Please provide saveFile!" )
            else:
                self.saveFile = saveFile
                self.outFile  = open( saveFile, 'wb' )
                self.outCSV   = csv.writer( self.outFile, quoting = csv.QUOTE_ALL )
        
        #   Initialize particles
        self.initializeParticles()
        self.initializeFitness()
        
        #   Create a checker for the 
        if maxGenNum:
            if type( maxGenNum ) == int:
                self.keepGoing = lambda x : ( x < maxGenNum )
            else:
                raise ValueError( "maxGenNum must be an integer yo" )
        else:
            self.keepGoing = lambda x : True
    
    
    def __del__( self ):
        """
        Close the file we opened for writing
        """
        self.outFile.close()
        
    
    def initializeParticles( self ):
        """
        Randomly distribute the positions and velocities in the range
        given by self.
        """
        xLow  = self.rangeSet[ :, 0 ]
        xHigh = self.rangeSet[ :, 1 ]
        xDif  = xHigh - xLow
        
        self.particleX = scipy.array( [  xLow +     xDif * r for r in scipy.rand( self.particleNumber, self.N ) ] )
        self.particleV = scipy.array( [ -xDif + 2 * xDif * r for r in scipy.rand( self.particleNumber, self.N ) ] )
        
        if self.saveVals:
            self.writeVals()
    
    
    def initializeFitness( self ):
        """
        Calculate the fitness for all of the given particle positions
        and record the positional and global best positions
        """
        self.bestFitPartX   = scipy.zeros( ( self.particleNumber, self.N ) )
        self.bestFitPart    = scipy.zeros( self.particleNumber )
        self.bestFitGlobalX = scipy.zeros( self.N )
        self.bestFitGlobal  = 0
        
        self.updateFitness()
    
    
    #-------------------------------#
    #   Algorithm update functions  #
    #-------------------------------#
    
    def takeStep( self ):
        """
        Take one step of the algorithm.  This consists of
            1.  Update the position
            2.  Update the velocity
            3.  Update the fitness values
            4.  Save to file if desired
        """
        self.updateVelocity()
        self.updatePosition()
        self.updateFitness()
        if self.saveVals:
            self.writeVals()
    
    
    def updateVelocity( self ):
        randDSet = scipy.rand( self.particleNumber )
        randGSet = scipy.rand( self.particleNumber )
        
        self.particleV = scipy.array( [  self.omega * self.particleV[ i ] + self.phiD * randDSet[ i ] * ( self.bestFitPartX[ i ] - self.particleX[ i ] ) + self.phiG * randGSet[ i ] * ( self.bestFitGlobalX - self.particleX[ i ] )  for i in range( self.particleNumber ) ] )
        
        
    def updatePosition( self ):
        """
        Save the current position as the "last" position, then take a 
        step dependent on the velocity
        """
        #print 'particleX       = {}'.format( self.particleX       )
        #print 'particleX.shape = {}'.format( self.particleX.shape )
        #print 'particleV       = {}'.format( self.particleV       )
        #print 'particleV.shape = {}'.format( self.particleV.shape )
        self.particleX += self.particleV
        
        
    def updateFitness( self ):
        """
        Calculate the new fitness values
        """
        for ( i, x ) in enumerate( self.particleX ):
            f = self.fitFunc( x )
            
            #   This particle
            if f < self.bestFitPart[ i ]:
                self.bestFitPart[ i ]  = f
                self.bestFitPartX[ i ] = x
            
            #   Globally
            if f < self.bestFitGlobal:
                self.bestFitGlobal  = f
                self.bestFitGlobalX = x
    
    
    def writeVals( self ):
        """
        Write the values to file -- flatten once, first
        """
        writeList = []
        for ( i, x ) in enumerate( self.particleX ):
            writeList += [ i ] + list( x )
        
        self.outCSV.writerow( writeList )
            
    
    #-------------------------------#
    #   Runnin and Gunnin           #
    #-------------------------------#
    
    def run( self ):
        """
        Take a step, and if it's supposed to be written, write!
        """
        step = 0
        
        while self.keepGoing( step ):
            self.takeStep()
            step += 1
