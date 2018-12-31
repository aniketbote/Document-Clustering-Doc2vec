# Document-Clustering-Doc2vec
Document Clustering Using Doc2vec method
# Dataset
Put your Dataset into the folder named as Articles   
Dataset type : The Dataset should contain text documents where        
1 document = 1 text file
# Stopwords
Document clustering is dependent on the words. However there are some words which are not important to the document itself and they contribute to the noice in data.         
They are called as stopwords. But some of these stopwords may be useless for one project but useful for another project. So depending on the project if you want to use your own customized stopword list, if there are some stopwords in default list that you don't want to use then place those words in stopwords_not_to_be_used.txt .
# Code
The code for clustering is in Clustering_code_Doc2Vec folder.     
The code is written in Python 3      
It uses K-means algorithm for clustering the documents.      
It uses the elbow method to find out the optimum value of K.       

When first executing the code execute by uncommenting the elbow method code. Find out the appropriate value of K.     

After getting the optimum value of K, comment the elbow method code and uncomment the K-means code, use the K value and you will get the required cluster values.     

# Required dependencies
Download the dependencies using:

pip install sklearn        
pip install nltk    
pip install gensim

