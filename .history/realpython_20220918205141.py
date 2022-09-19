############################################
##  Understanding Descriptive Statistics  ##
############################################

# descriptive statistics is about describing and summarizing data using two main approaches:
# the quantitative approach: describes and summarizes data numerically
# the visual approach: illustrates data with charts, plots, histograms, and other graphs

# when you describe and summarize a single variable, you're performing univariate analysis
# when you search for statistical relationships among a pair of variables, you're doing a bivariate analysis
# a multivariate analysis is concerned with multiple variables at once

#### TYPES OF MEASURES ####

# central tendency: tells you about the centers of the data
# useful measures include the mean, median, and mode
# variability: tells you about the spread of the data
# useful measures include variance and standard deviation
# correlation or join variability: tells you about the relation between a pair of varilables in a dataset
# useful measures include covariance and the correlation coefficient

#### POPULATION AND SAMPLES ####

# population is a set of all elements or items that you're interested in
# subset of a population is called a sample

#### OUTLIERS ####

# an outlier is a data point that differs significatly from the majority of the data taken from a sample or population
# possibly causes of outliers
# natural variation in data
# change in the behavior of the observed system
# errors in data collection


############################################
##  Choosing Python Statistics Libraries  ##
############################################

# Python’s statistics is a built-in Python library for descriptive statistics.
# You can use it if your datasets are not too large or if you can’t rely on importing other libraries.

# NumPy is a third-party library for numerical computing, optimized for working with single- and multi-dimensional arrays.
# Its primary type is the array type called ndarray.
# This library contains many routines for statistical analysis.

# SciPy is a third-party library for scientific computing based on NumPy.
# It offers additional functionality compared to NumPy, including scipy.stats for statistical analysis.

# Pandas is a third-party library for numerical computing based on NumPy.
# It excels in handling labeled one-dimensional (1D) data with Series objects and two-dimensional (2D) data with DataFrame objects.

# Matplotlib is a third-party library for data visualization.
# It works well in combination with NumPy, SciPy, and Pandas.


##########################################
##  Calculating Descriptive Statistics  ##
##########################################

# import packages
import matplotlib.pyplot as plt
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

# create some arbitrary numeric data to work with
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x

x_with_nan

# nan = not-a-number value
# used to replace missing values

math.isnan(np.nan), np.isnan(math.nan)

math.isnan(y_with_nan[3]), np.isnan(y_with_nan[3])

# creating more series objects that correspond to x and x_with_nan:
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y
y_with_nan
z
z_with_nan

# now have two NumPy arrays (y and y_with_nan)
# and two Pandas Series (z and z_with_nan)

####################################
##  Measures of Central Tendency  ##
####################################

# measures of central tendency: show the central or middle values of datasets
# several definitions of that including:
# mean
# weighted mean
# geometric mean
# harmonic mean
# median
# mode


#### MEAN ####

# sample mean (sample arithmetic mean or average)
# this is the pure mathematical way of calculating mean:
mean_ = sum(x) / len(x)
mean_

# Python has built-in statistics functions though:
mean_ = statistics.mean(x)
mean_

mean_ = statistics.fmean(x)
mean_

# however, if there are nan values among your data
# then statistics.mean() and statistics.fmean() will return nan as the output
mean_ = statistics.mean(x_with_nan)
mean_

mean_ = statistics.fmean(x_with_nan)
mean_

# if you use NumPy, then you can get the mean with np.mean()
mean_ = np.mean(y)
mean_

# about the mean() is a function, but you can also use .mean() as well
mean_ = y.mean()
mean_

# both the function 'mean()' and method '.mean()' from NumPy return the same results 'statistics.mean()'
# this will be nan if there are any nan values in your set
np.mean(y_with_nan)
y_with_nan.mean()

# if you need to ignore nan values, then use:
np.nanmean(y_with_nan)

# pd.Series objects also have the method '.mean()'
mean_ = z.mean()
mean_

#### WEIGHTED MEAN ####

# weighted mean (weighted arithmetic mean or weighted average) is a generalization of the arithmetic mean that enables you to define the relative contribution of each data point to the result
# ex:
0.2 * 2 + 0.5 * 4 + 0.3 * 8

# can also implement weight mean in pure Python by combining 'sum()' and either 'range()' or 'zip()'
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

# however, if you have large datasets, use 'np.average()'
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean

wmean = np.average(z, weights=w)
wmean

# can also calculate this element-wise product
(w * y).sum() / w.sum()

# but if the datasest contain nan values, it will produce nan
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)

np.average(z_with_nan, weights=w)

#### HARMONIC MEAN ####

# harmonic mean is the reciprocal of the mean of the reciprocals of all items in the dataset
# the pure mathematical version of finding this
hmean = len(x) / sum(1 / item for item in x)
hmean

# the python calculation of finding this
hmean = statistics.harmonic_mean(x)
hmean

statistics.harmonic_mean(x_with_nan)
statistics.harmonic_mean([1, 0, 2])
statistics.harmonic_mean([1, 2, -2])  # Raises StatisticsError

# this is the third way to calculate the harmonic mean
scipy.stats.hmean(y)
scipy.stats.hmean(z)

#### GEOMETRIC MEAN ####

# geometric mean is the n-th root of the product of all n elements x1 in a dataset x
# you can implement the geometric mean in pure Python like this:
gmean = 1
for item in x:
    gmean *= item
gmean **= 1 / len(x)
gmean

# there is also a new function to do the same thing
# it converts all values to floating-point numbers and returns their geometric mean
gmean = statistics.geometric_mean(x)
gmean

# If you pass data with nan values, then statistics.geometric_mean() will behave like most similar functions and return nan:
gmean = statistics.geometric_mean(x_with_nan)
gmean

# can also get the geometric mean with 'scipy.stats.gmean()':
scipy.stats.gmean(y)
scipy.stats.gmean(z)

#### MEDIAN ####

# sample median - the middle element of a sorted dataset
# can be sorted in increasing or decreasing order
# the mean is heavily affected by outliers but
# the median only depends on outliers either slightly or not at all

# pur Python implementations of the median
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])
median_

# the Python function for median:
median_ = statistics.median(x)
median_
median_ = statistics.median(x[:-1])
median_

# if the number of elements is even then there are two middle values
# 'median_low' returns the lower value
statistics.median_low(x[:-1])
# 'median_high' returns the higher value
statistics.median_high(x[:-1])

# the median functions don't return nan values even if they are in the dataset
statistics.median(x_with_nan)
statistics.median_low(x_with_nan)
statistics.median_high(x_with_nan)

# can also get median with 'np.median'
median_ = np.median(y)
median_
median_ = np.median(y[:-1])
median_

# you can use this to ignore all nan values
np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])

# the Pandas series objects have the method '.median()' that ignores nan values by default
z.median()
z_with_nan.median()


#### MODE ####

# sample mode is the value in the dataset that occurs most frequently
# if there isnt a single such value, then the set is multimodal (has multiple modal values)

# this is how to get mode with pure Python
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

# this is how to get mode with Python functions
mode_ = statistics.mode(u)  # returns a single value
mode_
mode_ = statistics.multimode(u)  # returns the multiple modes if applicable
mode_

# if you there is more than one modal value and you use 'mode()' then it will produce an error
v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v)

# these two handle nan values as regular
statistics.mode([2, math.nan, 2])
statistics.multimode([2, math.nan, 2])
statistics.mode([2, math.nan, 0, math.nan, 5])
statistics.multimode([2, math.nan, 0, math.nan, 5])

# can also use 'scipy.stats.mode()'
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_
mode_ = scipy.stats.mode(v)
mode_

# this function returns the object with the modal value and the number of times it occurs
# if multiple modal values in the dataset then only the smallest value is returned
mode_.mode
mode_.count

# Pandas Series objects have the method '.mode()' that handles multimodal values well and ignores nan values by default
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
v.mode()
w.mode()


#### MEASURES OF VARIABILITY ####
# measures of variability that quantify the spread of data points

#### VARIANCE ####
# sample variance quantifies the spread of the data

# this is how to calculate pure variance
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

# this is the Python function for variance
var_ = statistics.variance(x)
var_

# if nan values are among the data, then 'statistics.variance()' will return nan
statistics.variance(x_with_nan)

# can also calculate the sample variance with NumPy
var_ = np.var(y, ddof=1)
var_

var_ = y.var(ddof=1)
var_

# if there are nan values in the dataset, then np.var() and .var() will return nan
np.var(y_with_nan, ddof=1)
y_with_nan.var(ddof=1)

# if you want to skip nan values, use 'np.nanvar()'
np.nanvar(y_with_nan, ddof=1)

# pd.Series objects have the method '.var()' that skips nan values by default
z.var(ddof=1)
z_with_nan.var(ddof=1)

# to calculate population variance you would use 'statistics.pvariance()'

#### STANDARD DEVIATION ####

# the sample standard deviation is another measure of data spread

# to calculate standard deviation with pure Python
std_ = var_ ** 0.5
std_

# the Python function to do this is 'statistics.stdev()'
std_ = statistics.stdev(x)
std_

# if there are nan values in the dataset then it'll return nan
np.std(y, ddof=1)
y.std(ddof=1)
np.std(y_with_nan, ddof=1)
y_with_nan.std(ddof=1)
# to ignore nan values, you would use 'np.nanstd()'
np.nanstd(y_with_nan, ddof=1)

# pd.Series objects also have the method '.std()' that skips nan by default
z.std(ddof=1)
z_with_nan.std(ddof=1)

#### SKEWNESS ####

# sample skewness measures the asymmetry of a data sample
# skewness defined like this is called the adjsuted Fisher-Pearson standardized moment coefficient
# negative skew - indicates that there's a dominant tail on the left side
# positive skew - indivates that there's a dominant tail on the right side

# calculating sample skewness with Pure Python
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_

# calculate skewness with Python functions
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)

scipy.stats.skew(y_with_nan, bias=False)

# Pandas Series objects have the method '.skew()' that also returns the skewness of a dataset
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()

z_with_nan.skew()

#### PERCENTILES ####

# the sample p percentile is the element in the dataset such that p% of the elements in the dataset are less than or equal to that value
# each dataset has three quartiles which are percentiles that divide the dataset into four parts
# first quartile - is the sample 25th percentile (divides roughly 25% of the smallest items from the rest of the dataset)
# second quartile - the sample 50th percentile or the median
# third quartile - is the sample 75th percentile (divides roughly 25% of the largest items from the rest of the dataset)

# if you want to divide your data into several intervals use the 'statistics.quantiles()'
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)

statistics.quantiles(x, n=4, method='inclusive')

# you can also use 'np.percentile()' to determine any sample percentile in your dataset
y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)

# the percentile can be a sequence of numbers
np.percentile(y, [25, 50, 75])
np.median(y)

# if you want to ignore nan values, use 'np.nanpercentile()'
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan
np.nanpercentile(y_with_nan, [25, 50, 75])

# NumPy offers you very simple functionality in quantile() and nonquantile()
# need to provide the quantile values as the numbers between 0 and 1 instead of percentiles
np.quantile(y, 0.05)
np.quantile(y, 0.95)
np.quantile(y, [0.25, 0.5, 0.75])
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

# you need to pass 0.05 instead of 5 and 0.95 instead of 95
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)
z.quantile(0.95)
z.quantile([0.25, 0.5, 0.75])
z_with_nan.quantile([0.25, 0.5, 0.75])

#### RANGES ####

# the range of data is the difference between the maximum and minimum element in the dataset
np.ptp(y)
np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)

# the built-in Python, NumPy or Pandas functions and methods to calculate the maxima and minima of sequences
# max() and min() from the Python standard library
# amax() and amin() from NumPy
# nanmax() and nanmin() from NumPy to ignore nan values
# .max() and .min() from NumPy
# .max() and .min() from Pandas to ignore nan values by default

np.amax(y) - np.amin(y)
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
y.max() - y.min()
z.max() - z.min()
z_with_nan.max() - z_with_nan.min()

# the interquartile range is the difference between the first and third quartile
# once you calculate the quartiles, you can take their difference
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]

quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

#### SUMMARY OF DESCRIPTIVE STATISTICS ####

# get descriptive statistics with a single function or method call
result = scipy.stats.describe(y, ddof=1, bias=False)
result
# you can omit 'ddof=1' since its the default and only matters when calculating the variance
# can pass bias=False to force correcting the skewness and kurtosis for statistical bias

# describe() returns an object that holds the following descriptive statistics:
# nobs: the number of observations or elements in your dataset
# minmax: the tuple with the minimum and maximum values of your dataset
# mean: the mean of your dataset
# variance: the variance of your dataset
# skewness: the skewness of your dataset
# kurtosis: the kurtosis of your dataset

result.nobs
result.minmax[0]  # Min
result.minmax[1]  # Max
result.mean
result.variance
result.skewness
result.kurtosis

# Pandas has similar, if not better, functionality.
# Series objects have the method .describe()
result = z.describe()
result
# It returns a new Series that holds the following:
# count: the number of elements in your dataset
# mean: the mean of your dataset
# std: the standard deviation of your dataset
# min and max: the minimum and maximum values of your dataset
# 25%, 50%, and 75%: the quartiles of your dataset

result['mean']
result['std']
result['min']
result['max']
result['25%']
result['50%']
result['75%']

#### MEASURES OF CORRELATION BETWEEN PAIRS OF DATA ####

# the relationship between the corresponding elements of two variables in a dataset
# measures of correlation:
# Positive correlation exists when larger values of 𝑥 correspond to larger values of 𝑦 and vice versa.
# Negative correlation exists when larger values of 𝑥 correspond to smaller values of 𝑦 and vice versa.
# Weak or no correlation exists if there is no such apparent relationship.

# two statistics that measure the correlation between datasets are covariance and the correlation coefficient
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

#### COVARIANCE ####

# The sample covariance is a measure that quantifies the strength and direction of a relationship between a pair of variables:
# If the correlation is positive, then the covariance is positive, as well. A stronger relationship corresponds to a higher value of the covariance.
# If the correlation is negative, then the covariance is negative, as well. A stronger relationship corresponds to a lower (or higher absolute) value of the covariance.
# If the correlation is weak, then the covariance is close to zero.

# This is how you can calculate the covariance in pure Python:
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy

# NumPy has the function cov() that returns the covariance matrix:
cov_matrix = np.cov(x_, y_)
cov_matrix

# Note that cov() has the optional parameters bias, which defaults to False, and ddof, which defaults to None.
# Their default values are suitable for getting the sample covariance matrix.
# The upper-left element of the covariance matrix is the covariance of x and x, or the variance of x.
# Similarly, the lower-right element is the covariance of y and y, or the variance of y.
# You can check to see that this is true:
x_.var(ddof=1)
y_.var(ddof=1)

# The other two elements of the covariance matrix are equal and represent the actual covariance between x and y:
cov_xy = cov_matrix[0, 1]
cov_xy

cov_xy = cov_matrix[1, 0]
cov_xy

# Pandas Series have the method .cov() that you can use to calculate the covariance:
cov_xy = x__.cov(y__)
cov_xy

cov_xy = y__.cov(x__)
cov_xy

#### CORRELATION COEFFICIENT ####
# The correlation coefficient, or Pearson product-moment correlation coefficient, is denoted by the symbol 𝑟.
# The coefficient is another measure of the correlation between data.
# The value 𝑟 > 0 indicates positive correlation.
# The value 𝑟 < 0 indicates negative correlation.
# The value r = 1 is the maximum possible value of 𝑟. It corresponds to a perfect positive linear relationship between variables.
# The value r = −1 is the minimum possible value of 𝑟. It corresponds to a perfect negative linear relationship between variables.
# The value r ≈ 0, or when 𝑟 is around zero, means that the correlation between variables is weak.

# calculate the correlation coefficient with pure Python:
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

# scipy.stats has the routine pearsonr() that calculates the correlation coefficient and the 𝑝-value:
r, p = scipy.stats.pearsonr(x_, y_)
r
p

# you can apply np.corrcoef() with x_ and y_ as the arguments and get the correlation coefficient matrix:
corr_matrix = np.corrcoef(x_, y_)
corr_matrix

# The upper-left element is the correlation coefficient between x_ and x_.
# The lower-right element is the correlation coefficient between y_ and y_.
# Their values are equal to 1.0.
# The other two elements are equal and represent the actual correlation coefficient between x_ and y_:
r = corr_matrix[0, 1]
r

r = corr_matrix[1, 0]
r

# You can get the correlation coefficient with scipy.stats.linregress():
scipy.stats.linregress(x_, y_)

# linregress() takes x_ and y_, performs linear regression, and returns the results. slope and intercept define the equation of the regression line, while rvalue is the correlation coefficient.
# To access particular values from the result of linregress(), including the correlation coefficient, use dot notation:
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

# Pandas Series have the method .corr() for calculating the correlation coefficient:
r = x__.corr(y__)
r

r = y__.corr(x__)
r


############################
##  Working with 2D Data  ##
############################

#### AXES ####

# Start by creating a 2D NumPy array:
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a

# You can apply Python statistics functions and methods to it just as you would to 1D data:
np.mean(a)
a.mean()
np.median(a)
a.var(ddof=1)

# The functions and methods you’ve used so far have one optional parameter called axis, which is essential for handling 2D data. axis can take on any of the following values:

# axis = None says to calculate the statistics across all data in the array. The examples above work like this. This behavior is often the default in NumPy.
# axis = 0 says to calculate the statistics across all rows, that is , for each column of the array. This behavior is often the default for SciPy statistical functions.
# axis = 1 says to calculate the statistics across all columns, that is , for each row of the array.
# Let’s see axis = 0 in action with np.mean():

np.mean(a, axis=0)
a.mean(axis=0)
# The two statements above return new NumPy arrays with the mean for each column of a.
# In this example, the mean of the first column is 6.2.
# The second column has the mean 8.2, while the third has 1.8.

# If you provide axis=1 to mean(), then you’ll get the results for each row:
np.mean(a, axis=1)
a.mean(axis=1)

# The parameter axis works the same way with other NumPy functions and methods:
np.median(a, axis=0)
np.median(a, axis=1)
a.var(axis=0, ddof=1)
a.var(axis=1, ddof=1)

# This is very similar when you work with SciPy statistics functions.
# But remember that in this case, the default value for axis is 0:
scipy.stats.gmean(a)  # Default: axis=0
scipy.stats.gmean(a, axis=0)

# If you specify axis=1, then you’ll get the calculations across all columns, that is for each row:
scipy.stats.gmean(a, axis=1)

# If you want statistics for the entire dataset, then you have to provide axis=None:
scipy.stats.gmean(a, axis=None)

# You can get a Python statistics summary with a single function call for 2D data with scipy.stats.describe().
# It works similar to 1D arrays, but you have to be careful with the parameter axis:
scipy.stats.describe(a, axis=None, ddof=1, bias=False)
scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0
scipy.stats.describe(a, axis=1, ddof=1, bias=False)

# You can get a particular value from the summary with dot notation:
result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

#### DATAFRAMES ####

# Use the array a and create a DataFrame:
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df

# If you call Python statistics methods without arguments, then the DataFrame will return the results for each column:
df.mean()
df.var()

# If you want the results for each row, then just specify the parameter axis=1:
df.mean(axis=1)
df.var(axis=1)

# You can isolate each column of a DataFrame like this:
df['A']

# Now, you have the column 'A' in the form of a Series object and you can apply the appropriate methods:
df['A'].mean()
df['A'].var()

# It’s possible to get all data from a DataFrame with .values or .to_numpy():
df.values
df.to_numpy()

# Like Series, DataFrame objects have the method .describe() that returns another DataFrame with the statistics summary for all columns:
df.describe()
# The summary contains the following results:
# count: the number of items in each column
# mean: the mean of each column
# std: the standard deviation
# min and max: the minimum and maximum values
# 25%, 50%, and 75%: the percentiles

# You can access each item of the summary like this:
df.describe().at['mean', 'A']
df.describe().at['50%', 'B']


#######################
##  Visualizing Data ##
#######################

# matplotlib.pyplot is a very convenient and widely-used library, though it’s not the only Python library available for this purpose.
# You can import it like this:

plt.style.use('ggplot')

#### BOX PLOTS ####
# The box plot is an excellent tool to visually represent descriptive statistics of a given dataset.
# It can show the range, interquartile range, median, mode, outliers, and all quartiles.
# First, create some data to represent with a box plot:

np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

# Now that you have the data to work with, you can apply .boxplot() to get the box plot:

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

# The parameters of .boxplot() define the following:
# x is your data.
# vert sets the plot orientation to horizontal when False. The default orientation is vertical.
# showmeans shows the mean of your data when True.
# meanline represents the mean as a line when True. The default representation is a point.
# labels: the labels of your data.
# patch_artist determines how to draw the graph.
# medianprops denotes the properties of the line representing the median.
# meanprops indicates the properties of the line or dot representing the mean.

# You can see three box plots. Each of them corresponds to a single dataset (x, y, or z) and show the following:
# The mean is the red dashed line.
# The median is the purple line.
# The first quartile is the left edge of the blue rectangle.
# The third quartile is the right edge of the blue rectangle.
# The interquartile range is the length of the blue rectangle.
# The range contains everything from left to right.
# The outliers are the dots to the left and right.

#### HISTOGRAMS ####
# The histogram divides the values from a sorted dataset into intervals, also called bins.
# The values of the lower and upper bounds of a bin are called the bin edges.
# The frequency is a single value that corresponds to each bin.
# It’s the number of elements of the dataset with the values between the edges of the bin.

# If you divide a dataset with the bin edges 0, 5, 10, and 15, then there are three bins:
# The first and leftmost bin contains the values greater than or equal to 0 and less than 5.
# The second bin contains the values greater than or equal to 5 and less than 10.
# The third and rightmost bin contains the values greater than or equal to 10 and less than or equal to 15.

# The function np.histogram() is a convenient way to get data for histograms:
hist, bin_edges = np.histogram(x, bins=10)
hist

bin_edges

# It takes the array with your data and the number (or edges) of bins and returns two NumPy arrays:
# hist contains the frequency or the number of items corresponding to each bin.
# bin_edges contains the edges or bounds of the bin.

# What histogram() calculates, .hist() can show graphically:
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

# It’s possible to get the histogram with the cumulative numbers of items if you provide the argument cumulative=True to .hist():
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

# You can also directly draw a histogram with pd.Series.hist() using matplotlib in the background.

#### PIE CHARTS ####

# Pie charts represent data with a small number of labels and given relative frequencies.
# They work well even with the labels that can’t be ordered (like nominal data).

# Let’s define data associated to three labels:
x, y, z = 128, 256, 1024

# Now, create a pie chart with .pie():
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

#### BAR CHARTS ####
# Bar charts also illustrate data that correspond to given labels or discrete numeric values.
# They can show the pairs of data from two datasets.
# Items of one set are the labels, while the corresponding items of the other are their frequencies.
# Optionally, they can show the errors related to the frequencies, as well.
# The bar chart shows parallel rectangles called bars.
# Each bar corresponds to a single label and has a height proportional to the frequency or relative frequency of its label.
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

# You can create a bar chart with .bar() if you want vertical bars or .barh() if you’d like horizontal bars:
fig, ax = plt.subplots())
ax.bar(x, y, yerr = err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#### X-Y PLOTS ####
# The x-y plot or scatter plot represents the pairs of data from two datasets.
# The horizontal x-axis shows the values from the set x, while the vertical y-axis shows the corresponding values from the set y.
x=np.arange(21)
y=5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__=scipy.stats.linregress(x, y)
line=f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

# Then you can apply .plot() to get the x-y plot:
fig, ax=plt.subplots()
ax.plot(x, y, linewidth = 0, marker = 's', label = 'Data points')
ax.plot(x, intercept + slope * x, label = line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor = 'white')
plt.show()

#### HEATMAPS ####
# A heatmap can be used to visually show a matrix.
# The colors represent the numbers or elements of the matrix.
# Heatmaps are particularly useful for illustrating the covariance and correlation matrices.
# You can create the heatmap for a covariance matrix with .imshow():
matrix=np.cov(x, y).round(decimals = 2)
fig, ax=plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks = (0, 1), ticklabels = ('x', 'y'))
ax.yaxis.set(ticks = (0, 1), ticklabels = ('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha = 'center', va = 'center', color = 'w')
plt.show()
