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

# can also calculat the sample variance with NumPy
var_ = np.var(y, ddof=1)
var_

var_ = y.var(ddof=1)
var_
