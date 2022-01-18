'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    
    #điều chỉnh tìm ra theata
    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        for i in range(self.n_iter):  #thay xrange
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print ("Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta)
            # TODO:  add update equation here
            n,d = X.shape
            thetaDimensions,b = theta.shape     
            corrections = [0] * thetaDimensions  #tạo mảng phù hợp với số chiều của theta
            #công thức tính toán
            for j in range(0,n):
                for thetaDimension in range(0,thetaDimensions):
                    corrections[thetaDimension] += (theta.getT()*X[j,:].getT() - y[j])*X[j,thetaDimension]
            for thetaDimension in range(0,thetaDimensions):
                theta[thetaDimension] = theta[thetaDimension] - corrections[thetaDimension]*(self.alpha/n)               

        return theta
    
    #đánh giá theta thông qua hamf đánh giá hàm J()
    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''
        # TODO: add objective (cost) equation here
        n,d = X.shape
        cost = (X*theta - y).getT()*(X*theta - y)/(2*n)   #hàm đánh giá
        # print("Hello", cost[0,0])
        return cost[0,0]
    
    #getT() là tạo ra matraanj chuyển vị
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        n,d = X.shape
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))  #tạo ra ma trận 0 với d hàng 1 cột khởi tạo ma trận chứa
        self.theta = self.gradientDescent(X,y,self.theta)    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        # TODO:  add prediction function here
        return X*self.theta   # trả về hàm lý thuyết  y = theta0+ theta1*x1 + ... để đưa ra dự đoán cho y