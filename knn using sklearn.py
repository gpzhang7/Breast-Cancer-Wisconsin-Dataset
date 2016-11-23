import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
from random import randint



df = pd.read_csv('/Users/nickwalker/Desktop/Data Sets/Breast Cancer Wisconsin Data.csv')
df.replace('?',-99999,inplace=True)
    # columns with '?' have missing values -- so put in a -99,999 in order for them to be classified and not get an error
df.drop(['id'],1,inplace=True)
    # the id column is not something we need to feed into the classifier, so we drop it
    # note: that if you comment out this line the accuracy drops from ~96% to ~62%
    
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# create training and testing samples:
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

# define the classifier:
clf = neighbors.KNeighborsClassifier()

# train the classifier:
clf.fit(x_train,y_train)

# test and print the accuracy:
accuracy = clf.score(x_test,y_test)
print (accuracy)


# Next, we'll test random values to see what we get out
example_measures = np.array([[7,3,2,10,5,10,5,4,4],[4,2,1,1,1,2,3,2,1]])

example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)






