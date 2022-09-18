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

#
