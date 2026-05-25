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



# Question: Why do you want to shuffle the samples in the train and test datasets?





## Author

Nils Hornstein

