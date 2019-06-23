import pandas as pd
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
import re
import dill as pickle

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding='latin-1')
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding='latin-1')
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding='latin-1')
ratings.columns = ['userID', 'ISBN', 'bookRating']

combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)


combine_book_rating = combine_book_rating[combine_book_rating.bookRating != 0]

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

book_ratingCount = (combine_book_rating.
     groupby(by = ['userID'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalUserRatingCount'})
     [['userID', 'totalUserRatingCount']]
    )

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'userID', right_on = 'userID', how = 'left')

pd.set_option('display.float_format', lambda x: '%.3f' % x)

regularity_threshold = 50
rating_regular_user = rating_with_totalRatingCount.query('totalUserRatingCount >= @regularity_threshold')

combined = rating_regular_user.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)

us_canada_user_rating = us_canada_user_rating.drop(['Location','totalUserRatingCount'],axis=1)

ratings_df = us_canada_user_rating.drop(columns={'bookTitle'})

counts1 = ratings['userID'].value_counts() #No of ratings users have done
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)] #exclude users with less than 200 ratings

counts = ratings['bookRating'].value_counts()  #No of rating book has got
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)] #exclude book with less than 100 ratings

ratings_train_df, ratings_test_df = train_test_split(ratings_df,
                                   stratify=ratings_df['userID'], 
                                   test_size=0.20,
                                   random_state=42)

books_df = pd.Series(ratings_df.ISBN.unique())

#Indexing by personId to speed up the searches during evaluation
ratings_full_indexed_df = ratings_df.set_index('userID')
ratings_train_indexed_df = ratings_train_df.set_index('userID')
ratings_test_indexed_df = ratings_test_df.set_index('userID')

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

us_canada_user_rating = ratings_train_df.drop_duplicates(['userID', 'ISBN'])
# create pivot table i.e index = bookTitle and columns = UserId
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'userID', columns = 'ISBN', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)


model_knn_user = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn_user.fit(us_canada_user_rating_matrix)

class KNNUserRecommender:
    
    MODEL_NAME = 'KNN'
    
    def __init__(self,pd,model_knn, users=None, books=None,us_canada_user_rating=None,us_canada_user_rating_pivot=None):
        self.model_knn = model_knn
        self.us_canada_user_rating = us_canada_user_rating
        self.user_rating_pivot = us_canada_user_rating_pivot
        self.users = users
        self.books = books
        self.pd = pd
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def getSimilarUsers(self,user_id,query=[]):
        try:
            query_index = self.user_rating_pivot.index.get_loc(user_id)
        except KeyError:
            if(len(query) != 0):
                distances, indices = self.model_knn.kneighbors(query, n_neighbors = 100)
                users = []
                distance = []
                for i in range(0, len(distances.flatten())):
                    users.append(self.user_rating_pivot.index[indices.flatten()[i]])
                    distance.append(distances.flatten()[i])
                return self.pd.DataFrame({'userID':users,'Distance':distance}).sort_values(by='Distance', ascending=False)
            
            else:
                return self.pd.DataFrame({'userID':[],'Distance':[]})
        
        distances, indices = self.model_knn.kneighbors(self.user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 100)
        users = []
        distance = []
        for i in range(0, len(distances.flatten())):
            users.append(self.user_rating_pivot.index[indices.flatten()[i]])
            distance.append(distances.flatten()[i])
        
        return self.pd.DataFrame({'userID':users,'Distance':distance}).sort_values(by='Distance', ascending=False)
        
    def recommend_users(self, user_id, topn=10):
        # Recommend the more popular items that the user hasn't seen yet.
        self.popularity_df = self.getSimilarUsers(user_id)
        
        if(self.popularity_df.empty):
            return self.popularity_df
       
        recommendations_df = self.popularity_df
        
        return recommendations_df.head(topn)
    
    def predict_item_rating(self, user_id, book, topn=10):
        similiarUsers = list(self.recommend_users(user_id).userID)
        ratings = []
        for user in similiarUsers:
            ratings.extend(list(self.us_canada_user_rating.loc[self.us_canada_user_rating['userID'] == user].bookRating))
        return sum(ratings)/len(ratings)
    
    def recommend_items(self,user_id, books_to_ignore=[], topn=10, verbose=False):
        similiarUsers = list(self.recommend_users(user_id).userID)
        books = []
        for user in similiarUsers:
            books.extend(list(self.us_canada_user_rating.loc[self.us_canada_user_rating['userID'] == user].ISBN))
        
        recommendations_df = self.pd.DataFrame({'ISBN':list(set(books))})
        
        if verbose:
            if self.users is None:
                raise Exception('"books_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.books, how = 'left', 
                                                          left_on = 'ISBN', 
                                                          right_on = 'ISBN')[['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']]

        return recommendations_df.head(topn)
    
knn_user_model = KNNUserRecommender(pd,model_knn_user,users,books,us_canada_user_rating,us_canada_user_rating_pivot)

with open('pickles/knn_user_based_model.pickle','wb') as f:
    pickle.dump(knn_user_model, f)

pickle_in = open('pickles/knn_user_based_model.pickle','rb')
knn_user_model = pickle.load(pickle_in)

def save(fname, data):
    file = open(fname+".txt","w+")
    file.write(str(data))
    file.close()

from tqdm import tqdm

def get_books_rated(user_id, ratings_df):
    # Get the user's data and merge in the movie information.
    rated_books = ratings_df.loc[user_id]['ISBN']
    return set(rated_books if type(rated_books) == pd.Series else [rated_books])


#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:

    def get_not_rated_books_sample(self, user_id, sample_size, seed=42):
        rated_books = get_books_rated(user_id, ratings_full_indexed_df)
        all_books = set(books_df)
        non_rated_books = all_books - rated_books

        random.seed(seed)
        non_rated_books_sample = random.sample(non_rated_books, sample_size)
        return set(non_rated_books_sample)

    def _verify_hit_top_n(self, ISBN, recommended_books, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_books) if c == ISBN)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, user_id):
        #Getting the items in test set
        rated_values_testset = ratings_test_indexed_df.loc[user_id]
        if type(rated_values_testset['ISBN']) == pd.Series:
            user_rated_books_testset = set(rated_values_testset['ISBN'])
        else:
            user_rated_books_testset = set([int(rated_values_testset['ISBN'])])  
        rated_books_count_testset = len(user_rated_books_testset) 

        #Getting a ranked recommendation list from a model for a given user
        user_recs_df = model.recommend_items(user_id, 
                                               books_to_ignore=get_books_rated(user_id, 
                                               ratings_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for book_id in user_rated_books_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)    
            book_id = re.findall('\d+', book_id)
            if not book_id:
                book_id = random.randint(1111111111, 99999999999)
            else:
                book_id = book_id[0]
                
            non_rated_books_sample = self.get_not_rated_books_sample(user_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=book_id
                                                                          )

            #Combining the current interacted item with the 100 random items
            books_to_filter_recs = non_rated_books_sample.union(set([book_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = user_recs_df[user_recs_df['ISBN'].isin(books_to_filter_recs)]                    
            valid_recs = valid_recs_df['ISBN'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(book_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(book_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(rated_books_count_testset)
        recall_at_10 = hits_at_10_count / float(rated_books_count_testset)

        user_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'rated_count': rated_books_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return user_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, user_id in enumerate(tqdm(list(ratings_test_indexed_df.index.unique().values))):
            if idx % 100 == 0 and idx > 0:
                print('%d users processed' % idx)
            user_metrics = self.evaluate_model_for_user(model, user_id)  
            user_metrics['_user_id'] = user_id
            people_metrics.append(user_metrics)
        
            
        #detailed_results_df = user_metrics.sort_values(by=['col1'], ascending=False)
        detailed_results_df = pd.DataFrame(people_metrics).sort_values(by=['rated_count'], ascending=False)

        save("detailed_results_df",detailed_results_df)
        save("people_metrics",people_metrics)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['rated_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['rated_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()

print('Evaluating User Based recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(knn_user_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
file = open('result.txt', 'w+')
file.write(str(pop_global_metrics))
file.close()
print("Task Completed!!!")





