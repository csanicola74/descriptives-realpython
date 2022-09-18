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
