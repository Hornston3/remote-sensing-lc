# Question: What would you need to do to extract only the green and the infrared channel from this data?

Each 28 x 28 = 784 values of a row represent one channel. Assuming that green is the second channel, we would need to only keep or pick the columns 783 to 1567 of the data frame to extract the green channel. Note that we have to shift by one as data frame indices start by 0 instead of 1. The same applies for the infrared channel, we are assuming to be the last one. Here we would need to pick the columns 2351 up to 3134. 

# Question: What is the advantage of this encoding compared to a simple class label like '0', '1', '2', '3', '4', '5', or text labels like 'building', 'barren_land', ...?

Some ML algorithms require that all features are in a numerical format and can directly process a label like 'building'. Thereby, we need a way of converting categorical data. This could be done by simply encode it as an integer. This results in a certain degree of order, which may not be desirable or may even be detrimental to machine learning. For categorical variables where no such ordinal relationship exists, a one-hot encoding can be applied that does not imply any form of ordinal structure or distance between two categories.

# Question: Why use extend here and append above?

The key difference between both methods is that extend() adds all elements of a list individually to a list, while append() adds the whole list as a single element into the list. 

test_list.extend([5, 8, 12])
print(test_list)
[5, 8, 12]

test_list.append([5, 8, 12])
print(test_list)
[[5, 8, 12]]


# Question: What is wrong with the above code?

The problem with the code is that it is possible that one sample is present in both, the training as well as the test set. Thereby, our model might already know the (some) data of the test set and thus perform better. This would induce data leakage as we have directly "corrupted" that data as valid data set. We can not assure that the test set performance is a reliable measurement of generalisation of our model. 


# Question: Why do you want to shuffle the samples in the train and test datasets?

We simply just want to make sure that the model is not learning anything from the order of the data/samples. Maybe all samples of one class are grouped together or we have spatial ordering which could potential lead to biased training and thereby unreliable evalution of the models performance. Shuffle the samples prevents this issues and thus results in better generalization.

# Task 2.3

In the case where we only make use of the channels r, g, and b and drop the NIR channel, the overall classification accuracy is slightly going down. The classifier achieve roughly 4 % less accuracy than before. When taking a closer look at the specific per-class accuracies it can be observed that a overall decrease in accuracy is not the case for all classes. More interesting, the accuracy on the classes road and water is even rising in a comparable amount as it is decreasing for the other classes. 

# Task 2.4

In this second case where we only make use of the channels r, g, and nir and drop the b channel, the overall classification accuracy is slightly below our baseline. The classifier is roughly 1-2 % less accurate than before. Like in the first case, when taking a closer look at the specific per-class accuracies, it can be observed that a overall decrease in accuracy is not the case for all classes. Especially regarding the accuracy on the class water, we (nearly) achieve perfect accuracy. 

# Task 2.5 

For this task I picked the max_depth as hyperparameter. This denotes to the maximal depth of the tree and can be roughly described as the complexity of a tree. As default, the max_depth is set to None which result in a tree where all leaves are either pure or contrain less than min_samples_split samples (). In terms of expected behaviour when changing this hyperparameter with e.g. increasing depth, I assume that we will observe a rise in accuracy in the beginning. At some point, the classifier will reach a peak accuracy and will ongoing decrease due to under- and overfitting. Underfitting describes the phenomenon when we have an excessively simplistic formulation of the model. On the other hand, overfitting occurs when we have an excessively flexible and complex formulation. In this case, the model is more like memorizing the training data (including noise) and generalizes bad to the new unseen test data.

## Author

Nils Hornstein

