##  Project on Unsupervised Learning: Identify Customer Segments

### Project Overview
This project is part of the Udacity's Machine Learning Intro with PyTorch Nano Degree. 
The goal of the project is to work with real-life data provided by Bertelsmann partners AZ Direct and Arvato Financial Solutions. They have provided two datasets one with demographic information about the people of Germany, and one with that same information for customers of a mail-order sales company. The data here concerns a company that performs mail-order sales in Germany. Their main question of interest is to look at relationships between demographics features, organize the population into clusters, and see how prevalent customers are in each of the segments obtained and to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. As a data scientist, the main objective in this project is to use unsupervised Machine Learning techniques to organize the general population into clusters, then use those clusters to see which of them comprise the main user base for the company. Prior to applying the machine learning methods, data assessment, cleaning and feature transformation to be applied in order to convert the data into a usable form.

### Project Motivation
The unsupervised learning branch of machine learning is key in the organization of large and complex datasets. While unsupervised learning lies in contrast to supervised learning in the fact that unsupervised learning lacks objective output classes or values, it can still be important in converting the data into a form that can be used in a supervised learning task. Dimensionality reduction techniques can help surface the main signals and associations in the data, providing supervised learning techniques a more focused set of features upon which to apply their work. Clustering techniques are useful for understanding how the data points themselves are organized. These clusters might themselves be a useful feature in a directed supervised learning task. This project gives hands-on experience with a real-life task that makes use of these techniques, focusing on the unsupervised work that goes into understanding a dataset.

In addition, the dataset presented in this project requires a number of assessment and cleaning steps before it can be applied to the machine learning methods. The ability to perform data wrangling and the ability to make decisions on data that are worked in this project, are both valuable skills in the data science and machine learning fields.

### File Descriptions
* Identify_Customer_Segments.ipynb: This is the project Jupyter Notebook file containng the code project work, divided into sections and guidelines for completing the project. 
* Identify_Customer_Segments.html: html version of the Jupyter notebook file.
* Udacity_AZDIAS_Subset.csv: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
* Udacity_CUSTOMERS_Subset.csv: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
* AZDIAS_Feature_Summary.csv: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
* Data_Dictionary.md: Detailed information file about the features in the provided datasets.
* readme.md: Project readme file containing the project details and resources info

**Note: Based on the Udacity project agreement, the csv files/datasets are removed after the project being completed.**


### Installation and Libraries

This project requires **Python 3.7** and the following Python libraries :

- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)


### Licensing, Authors, Acknowledgements
* Author: [Sidharth Sabat](https://www.linkedin.com/in/sidharthsabat88/)
* Thanks to Udacity and Arvato for the datasets. Based on the project agreement, all datasets are removed from this repository.