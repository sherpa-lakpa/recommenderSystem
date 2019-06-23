class PopularityRecommender:
        MODEL_NAME = 'Popularity'
        
        def __init__(self, popularity_df, books_df=None):
            self.popularity_df = popularity_df
            self.books_df = books_df
            
        def get_model_name(self):
            return self.MODEL_NAME
            
        def recommend_items(self, user_id, books_to_ignore=[], topn=10, verbose=False):
            # Recommend the more popular items that the user hasn't seen yet.
            recommendations_df = self.popularity_df[~self.popularity_df['ISBN'].isin(books_to_ignore)] \
                                   .sort_values('bookRating', ascending = False) \
                                   .head(topn)

            if verbose:
                if self.books_df is None:
                    raise Exception('"books_df" is required in verbose mode')

                recommendations_df = recommendations_df.merge(self.books_df, how = 'left', 
                                                              left_on = 'contentId', 
                                                              right_on = 'contentId')[['bookRating', 'ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlL']]
            return recommendations_df