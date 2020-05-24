# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:09:24 2020

@author: Gowrisankar JG
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import pandas as pd
from scipy.sparse import csr_matrix

df = pd.read_csv('wikipedia-vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())
titles = list(df.columns)

svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)
pipeline.fit(articles)
labels = pipeline.predict(articles)
df = pd.DataFrame({'label': labels, 'article': titles})
print(df.sort_values('label'))

#####################################################################################

from sklearn.decomposition import NMF

model = NMF(n_components=6)
model.fit(articles)
nmf_features = model.transform(articles)
print(nmf_features)
df = pd.DataFrame(nmf_features, index=titles)
print(df.loc['Anne Hathaway'])
print(df.loc['Denzel Washington'])






























