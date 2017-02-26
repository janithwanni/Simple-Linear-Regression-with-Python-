from numpy import zeros, ones, array, linspace, logspace ,linalg
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
        #self.plotData(self.X,self.Y)
        #initialize theta
        theta = np.matrix('0;0') #you could just use the function zeros if needed
        #calculate cost with theta 0,0 just to be safe that the function works
        self.X = np.array(self.X,dtype=np.float)
        self.Y = np.array(self.Y,dtype=np.float)
        self.Y = np.transpose(np.matrix(self.Y))
        self.X = np.matrix(np.transpose(np.vstack((ones((len(self.X))),self.X))))
        #optional print suppressor turn off when needed
        #np.set_printoptions(precision=4,suppress=True)
        print("Cost with theta set to 0",self.calculateCost(theta,0))
        theta = self.optimizeTheta(theta,0.000001,0,200)
        print("Theta found after 200 Iterations and lamda as 0\n",theta)
        print("Cost with optimized theta and lamda set to 0",self.calculateCost(theta,0))
        #self.plotCurve(theta)
        theta2 = self.normalEquation(theta,0)
        print("Theta found with normal equation\n",theta2)
        print("Cost with theta found from normalequation",self.calculateCost(theta2,0))
        self.plotCurve(theta)
    def calculateCost(self,theta,lamda):
        #added lambda as well for the regularization
        #print("cost")
        h = self.X * theta;
        res = h - self.Y;
        J = np.sum(np.power(res,2))/ (2 * len(self.X))
        reg = (lamda * np.power(np.sum(theta[1:]),2)) / (2 * len(self.X))
        J += reg
        return J
    def optimizeTheta(self,theta,alpha,lamda,numOfIter):
        print("calculating gradient")
        h = self.X * theta
        Xt = (np.transpose(self.X))
        hm = h - self.Y
        grad = Xt * hm
        grad *= (alpha /len(self.X))
        prevCost = self.calculateCost(theta,lamda)
        for i in range(1,numOfIter):
            theta = theta - grad
            if(self.calculateCost(theta,lamda) > prevCost):
                break
        return theta    
    def normalEquation(self,theta,lamda):
        print("Normal method")
        theta = linalg.inv(np.transpose(self.X) * self.X) * np.transpose(self.X) * self.Y
        return theta
    def plotData(self,x,y):
        #plot data to show the graph of brain weight and body weight
        print("Preparing to Plot Data")
        plt.scatter(x,y)
        plt.xlabel('Brain Weight in grams')
        plt.ylabel('Body Weight in grams')
        plt.title('Data of different mammals on brain and body weight')
        plt.grid(True)
        plt.savefig('yes.png')
        plt.show()
        print("Plotting Data")

    def plotCurve(self,theta):
        print("Preparing to plot curve using the theta after gradient descent or normal equation")
        plt.scatter(self.X[:,1:],self.Y)
        plt.xlabel('Brain Weight in grams')
        plt.ylabel('Body Weight in grams')
        plt.title('Data of different mammals on brain and body weight')
        plt.grid(True)
        #print(self.X)
        #print(theta)
        line = plt.plot(self.X[:,1:],self.X * theta)
        plt.setp(line,'color','r')
        plt.savefig('yes2.png')
        plt.show()
    def getinputs(self,filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        lines.pop() #take out the last new line
        #indexes = []
        X_inputs = []
        Y_inputs = []
        for word in lines:
            #word is a string fixed in the following format
            #XX__XXXX.XXX__XXXX.XX
            #indexes.append(int(word[0:2].strip()))
            X_inputs.append(float(word[4:11].strip()))
            Y_inputs.append(float(word[14:].strip()))
        self.X = X_inputs
        self.Y = Y_inputs
        print("Collected inputs")
linreg = LinRegBrainWeight()

