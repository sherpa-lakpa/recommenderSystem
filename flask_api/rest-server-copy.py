#!flask/bin/python
import os
import pandas as pd
import numpy as np
import dill as pickle
import builtins
from flask import Flask, jsonify, abort, request, make_response, url_for

app = Flask(__name__, static_url_path = "")

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

@app.route('/api/content/<ISBN>', methods = ['GET'])
def get_content(ISBN):
    pickle_in = open('pickles/knn_item_model.pickle','rb')
    knn_item_model = pickle.load(pickle_in)
    user_recs_df = knn_item_model.recommend_items(ISBN,topn=10, verbose=True)
    responses = user_recs_df.to_json(orient="records")
    return (responses)


@app.route('/api/user/<int:user_id>', methods = ['GET'])
def get_user(user_id):
    pickle_in = open('pickles/knn_user_model.pickle','rb')
    knn_user_model = pickle.load(pickle_in)

    pickle_datain = open('pickles/us_canada_user_rating.pickle','rb')
    us_canada_user_rating = pickle.load(pickle_datain)
    user_recs_df = knn_user_model.recommend_items(user_id,topn=10, verbose=True)
    responses = user_recs_df.to_json(orient="records")

    pickle_in.close()
    del pickle_in
    pickle_datain.close()
    del pickle_datain

    del knn_user_model
    del us_canada_user_rating
    del user_recs_df
    return (responses)



@app.route('/api/popular/<int:user_id>', methods = ['GET'])
def get_popular(user_id):
    pickle_in = open('pickles/popularity_model.pickle','rb')
    popularity_model = pickle.load(pickle_in)
    user_recs_df = popularity_model.recommend_items(user_id, 
                                    topn=10, verbose=True)
    responses = user_recs_df.to_json(orient="records")
    pickle_in.close()
    del pickle_in
    del popularity_model
    del user_recs_df
    return (responses)


@app.route('/api/similiar_user/<int:user_id>', methods = ['GET'])
def get_similiarUser(user_id):
    pickle_in = open('pickles/knn_user_model.pickle','rb')
    knn_user_model = pickle.load(pickle_in)
    pickle_in = open('pickles/us_canada_user_rating.pickle','rb')
    us_canada_user_rating = pickle.load(pickle_in)
    user_recs_df = knn_user_model.getSimilarUsers(user_id)
    responses = user_recs_df.to_json(orient="records")
    return (responses)
