import pandas as pd
import numpy as np
import pickle
import re
from scipy.sparse import csr_matrix

knnmodel = pickle.load(open('knn-model','rb'))
pickle.dump(knnmodel,open('knn-model','wb'))
df=pd.read_csv('Final_Ratings.csv',nrows=142012)
df.head()

book_pivot = df.pivot_table(columns='userId',index='bookTitle',values='bookRating')
book_pivot.fillna(0,inplace=True)
book_sparse=csr_matrix(book_pivot)


def recommendation_knn(book_name):
    recommend_books = []
    book_id = np.where(book_pivot.index==book_name)[0][0]
    distances,suggestions=knnmodel.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors = 11)    
    
    books=[]
    for i in range(len(suggestions)):
        if i==0:
            print("The suggestions for ",book_name,"are : ")
        if not i:
            books = book_pivot.index[suggestions[i]]
    for i in range(1,len(books)):
         recommend_books.append(books[i] )
    return recommend_books 