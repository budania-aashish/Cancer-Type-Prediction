import numpy as np 
from math import sqrt 
import warnings  
from collections import Counter 
import pandas as pd 
import random #to shuffle the dataset randomly 


def k_nearest_neighbors(data,predict,k):#function of knn 
    if len(data)>=k:
        #check for the valid value of 
        warnings.warn("Yeah! You know that k should be less than or equal to total voting groups")

    distances=[]  #declaration of array to store the data
    for group in data: #select a group or we can say kind of index for one line of data 
        for features in data[group]:
            #for all of the feautures of the data 
            #inbuilt function to do calculations for euclidean distance for any number of dimensions 
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #appending the result into distance array 
            distances.append([euclidean_distance,group]) 
            
    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = (Counter(votes).most_common(1)[0][1])*1.0/k
    #print(vote_result , confidence)
    return vote_result , confidence #here are the two class lables we are returning one of them           

#reading the file containing the dataset     
df = pd.read_csv("breast_cancer.data")

#changing the data values that is missing from the file 
#we can understand the meaning of -99999 based on the value of k 
df.replace('?',-999999,inplace=True)

#id column isn't required and also it has very uncommon values of data set  
df.drop(['id'],1,inplace=True)

#convert all the data in float values most probably we read in the type of strings from csv file 
full_data=df.astype(float).values.tolist()

#full_data is the dataset as list of lists 
random.shuffle(full_data)

test_size = 0.2  #fraction to train 
train_set = {2:[] , 4:[]}#dictionary 
test_set = {2:[] , 4:[]}#dictionary 

#getting an index for the dataset that is used in train data set 
train_data = full_data[:-int(test_size*len(full_data))]

#this is last 20% of the data 
test_data = full_data[-int(test_size*len(full_data)):]

#populate the dictionaries 
for i in train_data: 
    train_set[i[-1]].append(i[:-1])#for last column either 2 or 4  

for i in test_data: 
    test_set[i[-1]].append(i[:-1])

correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbors(train_set , data ,k =13)
        if group == vote: #in case of trained class label and test class label are same  
            correct +=1.0 
        total +=1.0
ratio = correct/total

print '\nAccuracy of the KNN Algorithm is \n',ratio
