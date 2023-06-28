import numpy as np
import pandas as pd


def jaccard_coef(x, y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    difference = intersection.sum() / float(union.sum())
    return difference


# Import training data - contained in Trainingdataset.xlsx
datatrain = pd.read_excel(
    'D:\School\Data Mining\Project 1\Trainingdataset.xlsx')
# Import testing data - contained in Testingdataset.xlsx
datatest = pd.read_excel('D:\School\Data Mining\Project 1\Testingdataset.xlsx')
# Print the total number of data entries in dataset- should be 200 in training set
print('Number of Data entries in training set:',len(datatrain))
# Print the total number of data entries in testing dataset- should be 19
print('Number of Data entries in training set:',len(datatest))
# Include all rows, exclude columns 0 (names) and column 1 (actual rating), set equal to x, this will be used for data manipulation
clean_train = datatrain.iloc[:, 2:]
print('Cleaned training data:', len(clean_train), 'entries -this should still be 200')
print(clean_train.head())
# Include all rows, exclude columns 0 (names) and column 1 (classification), set equal to Y, this will be used for data manipulation
clean_test = datatest.iloc[:, 2:]
print('Cleaned testing data:', len(clean_test),'entries- this should still be 19')
print(clean_test.head())


# Flattening training data frame row into one flat list
row_list = clean_train.loc[0,:].values.flatten().tolist()
# print(row_list)
# Flattening testing data frame row into one flat list
row2_list = clean_test.loc[1, :].values.flatten().tolist()
# print(row2_list)


# Calling Jaccard function
simtest = jaccard_coef(row_list, row2_list)
print('Jaccard Coeffecient:', simtest)
