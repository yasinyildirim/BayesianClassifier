import math
import numpy as np

#bayes classifier class
class GaussianBayesClassifier:
    def __init__(self):
        return
    def calc_mean(self, data):
        mean_value = sum(data) / float(len(data))
        return mean_value
    def calc_standart_deviation(self, data):
        mean = self.calc_mean(data)
        variance = sum([pow(ith_value-mean,2) for ith_value in data])/float(len(data)-1)
        return math.sqrt(variance)
    def calc_standart_deviation(self, data, mean):
        variance = sum([pow(ith_value-mean,2) for ith_value in data])/float(len(data)-1)
        return math.sqrt(variance)
    def calculateProbability(self, x, mean, standard_deviation):
		#calculate probabilty of given x value using Gaussian distribution
        exponential_part = math.exp(-(math.pow(x-mean,2)/(2*math.pow(standard_deviation,2))))
        factor_part = (1 / (math.sqrt(2*math.pi) * standard_deviation));
        return factor_part * exponential_part
    def calc_2d_covariance_matrix(self, data1, data2, mean1, mean2, variance1, variance2):
        covariance_matrix = np.zeros((2, 2))
        covariance_matrix[0][0] = variance1
        covariance_matrix[1][1] = variance2
        covariance_matrix[0][1] = sum([(ith_value1-mean1)*(ith_value2-mean2)
          for ith_value1, ith_value2  in zip(data1, data2)])/float(len(data1)-1)
        covariance_matrix[1][0] = covariance_matrix[0][1]
        return covariance_matrix
    def calc_covariance_matrix(self, data_vector, mean_vactor, stddev_vector):
        covariance_matrix = np.zeros((mean_vactor.size(), mean_vactor.size() ))
        for i in range(0, mean_vector.size()):
            for j in range(0, mean_vector.size()):
                if(i==j):
                    covariance_matrix[i][j] =  stddev_vector[0][j]**2;
                elif( covariance_matrix[i][j] == 0):
                    covariance_matrix[i][j] = sum([(ith_value1-mean1)*(ith_value2-mean2)
                     for ith_value1, ith_value2  in zip(data_vector[0][i], data_vector[0][j])])/float(len(data1)-1)
                    #it is diagonal matrix
                    covariance_matrix[j][i] = covariance_matrix[i][j] 
    def getClassDistribution(self, class_features):
        params = {}
        means = []
        sigmas = []
        #for each feature calculate the mean and standard deviation
        # (in our case features are x, and y coordinate values) 
        for feature in class_features:
            mean = self.calc_mean(feature)
            sigma = self.calc_standart_deviation(feature, mean)
            means.append(mean)
            sigmas.append(sigma)
        params[0] = means
        params[1] = sigmas
        return params
    #input_data represents input features    
    def calculateClassProbabilities(self, input_data, class_distributions):
        probabilities = {}
        for class_key, classDistribution in class_distributions.iteritems():
            probabilities[class_key] = 1
            #According to bayesian classification multiplication of conditional 
            #probabilites (likelihoods) of independent features yields a posteriori probability of that class
            for i in range(len(classDistribution)):
                mean = classDistribution[i]
                x = input_data[i]
                # multiply likelihoods of features
                probabilities[class_key] *= self.calculateProbability(x, mean, stdev)
        return probabilities
    #def calculateDecisionBoundary(class_distributions):
 
