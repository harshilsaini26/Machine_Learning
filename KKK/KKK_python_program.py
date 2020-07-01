import math
import operator
import csv
import random

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(len(dataset)-1):
            for j in range(4):
                dataset[i][j] = float(dataset[i][j])
            if random.random() < split:
                trainingSet.append(dataset[i])
            else:
                testSet.append(dataset[i])

def euclid_dist(val1, val2, length):
    distance = 0
    for i in range(length):
        distance += pow((val1[i] - val2[i]),2)
    return math.sqrt(distance)

def Neighbours(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance)-1
    for i in range(len(trainingSet)):
        dist = euclid_dist(testInstance, trainingSet[i], length)
        distance.append((trainingSet[i],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbour = []
    for i in range(k):
        neighbour.append(distance[i][0])
    return neighbour

def Response(neighbour):
    class_votes = {}
    for i in range(len(neighbour)):
        response = neighbour[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

def Accuracy(testSet, prediction):
    counter = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == prediction[i]:
            counter += 1
    print(counter)
    return (counter/float(len(testSet)))*100


if __name__ == '__main__':
    trainingSet = []
    testSet = []
    split = 0.70
    
    # enter the data location
    
    loadDataset(r'', split, trainingSet, testSet)
    print('Train Set : '+ str(len(trainingSet)))
    print('Test Set : '+ str(len(testSet)))
    prediction = []
    
    # set your own 'k' values
    
    k = 
    for x in range(len(testSet)):
        neighbour = Neighbours(trainingSet, testSet[x], k)
        result = Response(neighbour)
        prediction.append(result)
    acc = Accuracy(testSet, prediction)
    print('Accuracy : '+str(acc)+' %')
   
