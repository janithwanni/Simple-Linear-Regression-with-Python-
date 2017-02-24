from numpy import zeros, ones, array, linspace, logspace
import numpy as np
import matplotlib.pyplot as plt


class LinRegBrainWeight:
    'Linear regression model to predict Brain Weight and Body Weight.The data records the average weight of the brain and body for a number of mammal species.'
    X = []
    Y = []
    def __init__(self):
        #begin assembling and plotting and calculating cost
        #and minimizing parameteres and finally predict
        self.getinputs('x01.txt')
        #plot data if necessary
        #self.plotData()
        #initialize theta
        theta = np.matrix('0;0') #you could just use the function zeros if needed
        #calculate cost with theta 0,0 just to be safe that the function works
        self.X = np.array(self.X,dtype=np.float)
        self.X = np.matrix(np.transpose(np.vstack((ones((len(self.X))),self.X))))
        #optional print suppressor turn off when needed
        np.set_printoptions(precision=4,suppress=True)
        print("Cost with theta set to 0",self.calculateCost(theta,0))
        
    def calculateCost(self,theta,lamda):
        #added lambda as well for the regularization
        print("cost")
        h = self.X * theta;
        res = h - self.Y;
        J = np.sum(res**2)/ (2 * len(self.X))
        reg = (lamda * np.sum(theta[1:] ** 2)) / (2 * len(self.X))
        J += reg
        return J
    def optimizeTheta(self,theta,lamda,numOfIter):
        print("calculating gradient")
        h = self.X * theta
        grad = np.transpose(self.X) * (h - y) * (alpha /m)
        for i in range(1,numOfIter):
            theta = theta - grad
            
    def minimizeCost(self,alpha,noOfIter):
        print("minimizing")
        
    def normalEquation(self,theta,lamda):
        print("Normal method")
    def plotData(self):
        #plot data to show the graph of brain weight and body weight
        print("Preparing to Plot Data")
        plt.scatter(self.X,self.Y)
        plt.xlabel('Brain Weight in grams')
        plt.ylabel('Body Weight in grams')
        plt.title('Data of different mammals on brain and body weight')
        plt.grid(True)
        plt.savefig('yes.png')
        plt.show()
        print("Plotting Data")
        
    def getinputs(self,filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        lines.pop() #take out the last new line
        indexes = []
        X_inputs = []
        Y_inputs = []
        for word in lines:
            #word is a string fixed in the following format
            #XX__XXXX.XXX__XXXX.XX
            indexes.append(int(word[0:2].strip()))
            X_inputs.append(float(word[4:11].strip()))
            Y_inputs.append(float(word[14:].strip()))
        self.X = X_inputs
        self.Y = Y_inputs
        print("Collected inputs")
linreg = LinRegBrainWeight()

