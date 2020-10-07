# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:41:11 2020

@author: Wendy
"""

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from surprise import NMF, Dataset, Reader

# Unpickle
with open('pickles/df_small_reduced.pkl', 'rb') as f_dfsmall:
    df_small = pickle.load(f_dfsmall)
    
with open('pickles/df_expand_genres_reduced.pkl', 'rb') as f_dfgenres:
    df_expand_genres = pickle.load(f_dfgenres)
    
with open('pickles/genres.pkl', 'rb') as f_genres:
    genres = pickle.load(f_genres)
    
with open('pickles/features_vectorized.pkl', 'rb') as f_vecfeatures:
    features_vectorized = pickle.load(f_vecfeatures)
    
with open('pickles/ratings.pkl', 'rb') as f_ratings:
    ratings = pickle.load(f_ratings)
    

# Add the option 'None' to the list with genres
genres_None = [None]    
genres = genres_None + genres

# Define function to calculate the cosine similarity and store in cache.
@st.cache
def cosine(features_vectorized):
    csim = cosine_similarity(features_vectorized)
    return csim

# Create a dataframe with the cosine similarity between movies based on their
# vectorized features
csim_features = cosine(features_vectorized)
csim_features = pd.DataFrame(csim_features, columns=df_small.title,\
                             index=df_small.title)


# Define function for making general recommendations 
def general_recommendation(genre=None, original_language=None, quantity=10, ):
    
    '''Function that will recommend movies based on their weighted vote average,
    with the option to filter by genre and original language.
    
    Parameters
    ----------------
    - genre: str
    - original_language: str
    - quantity: int, quantity of recommendations to return
    
    '''
    df_small_sorted = df_small.sort_values('weighted_vote_average',\
                                           ascending=False)
    df_expand_genres_sorted = df_expand_genres.sort_values('weighted_vote_average',\
                                                           ascending=False)
    
    if genre == None:
        if original_language == None:
            df = df_small_sorted[['title','weighted_vote_average']].head(quantity)
        else:
            df = df_small_sorted.loc[(df_small_sorted.original_language ==\
                                      original_language)]
            df = df[['title','weighted_vote_average']].head(quantity)
    
    else:
        if original_language == None:                                     
            df = df_expand_genres_sorted.loc[(df_expand_genres_sorted.genres_list ==\
                                              genre)]
            df = df[['title','weighted_vote_average']].drop_duplicates().head(quantity)
        else:                                         
            df = df_expand_genres_sorted.loc[\
                    (df_expand_genres_sorted.original_language == original_language)\
                    &(df_expand_genres_sorted.genres_list == genre)]
            df = df[['title','weighted_vote_average']].drop_duplicates().head(quantity)
    
    df.index = np.arange(1, len(df) + 1)
    
    return df
   

# Define function for making recommendations based on similarity of movie 
# features.
def recommendation_content(title, quantity=10):
    
    '''Function that takes an input movie and recommends movies with similar
    features as the input movie.
    
    Parameters
    ----------------
    - title: str, title of the input movie
    - quantity: int, quantity of recommendations to return
    
    '''
            
    df_similar_features = pd.DataFrame(csim_features.loc[title].sort_values\
                                       (ascending=False)).reset_index()
    df_similar_features.columns = ['title', 'cosine_sim']
    df_similar_features = df_similar_features.merge(df_small[['title',\
                                        'weighted_vote_average']], on='title')
    df_similar_features = df_similar_features.drop(df_similar_features\
                            [df_similar_features.title == title].index)
    df_similar_features = df_similar_features.drop(df_similar_features\
                            [df_similar_features.weighted_vote_average<7].index)
    
    df_similar_features.index = np.arange(1, len(df_similar_features) + 1)
        
    return df_similar_features[['title', 'weighted_vote_average']].head(quantity)



# Define function for making recommendations based on collaborative filtering,
# using the NMF model of the library Surprise
def recommendation_nmf(my_ratings, n=10):
    
    '''This functions takes a list of user ratings as an input and 
    returns a recommendation comparing these ratings with other users 
    ratings in the dataset. It uses an NMF algorithm.'''
    
    
    ratings2 = ratings.append(my_ratings)
    reader = Reader(rating_scale=(ratings2["rating"].min(),ratings2["rating"].max()))
    data = Dataset.load_from_df(ratings2,reader)
    
    model_NMF = NMF(n_factors=10, n_epochs=100, biased=False, reg_pu=0.1,\
                    reg_qi=0.1, random_state=4)
    trainset = data.build_full_trainset()
    model_NMF.fit(trainset)
    
    testset = trainset.build_anti_testset()
    predictions = model_NMF.test(testset[-9066:-1])
    
    userid = max(ratings2.userId)
    top_n = []
    
    for uid, iid, true_r, est, _ in predictions:
        if uid==userid:
            top_n.append((iid, est))
        
    top_n.sort(key=lambda x: x[1], reverse=True)
    top_n = pd.DataFrame([x[0] for x in top_n], columns=['movieId'])
    top_n = top_n.merge(df_small[['title', 'movieId', 'weighted_vote_average']],\
                        on='movieId')
    top_n = top_n.drop(top_n[top_n.weighted_vote_average<7].index)
    
    top_n.index = np.arange(1, len(top_n) + 1)
    
    return top_n[['title', 'weighted_vote_average']].head(n)



# Create sidebar with types of recommendation
tipo = st.sidebar.selectbox(label='Tipo de recomendación', 
                            options= [None,
                                      'Recomendación general', 
                                      'Recomendación por contenido',
                                      'Recomendación por filtro colaborativo'])

# If the option 'None' is selected, execute this code
if tipo == None:
    
    st.title('Recomendación de Películas')
    
    st.write('')
    
    st.write('Este sistema de recomendación de películas fue creado como\
             proyecto integrador para el curso de Data Science en Digital\
             House.')
    
    st.write('En el menú desplegable, de la izquierda, se puede elegir un\
             tipo de recomendación:')
    
    st.markdown('- **Recomendación general:** Ofrece la opción de elegir un\
                género e idioma original, y devuelve una recomendación de 10\
                películas ordenadas por su calificación ponderada.')
    st.markdown('- **Recomendación por contenido:** Eligiendo una película,\
                usa la similitud coseno para\
                devolver una recomendación de 10 películas similares a la\
                película de entrada (tomando en cuenta los géneros, actores\
                principales, directores y compañías de producción).')
    st.markdown('- **Recomendación por filtro colaborativo:** Devuelve una\
                recomendación de 10 películas comparando tus preferencias\
                con los ratings de otros usuarios. Utiliza un algoritmo de\
                *non-negative matrix factorization* (NMF).')
    
    st.markdown('La recomendación general y la recomendación por contenido\
                se basan en un dataset con metadata de 9066 películas. La\
                recomendación por filtro colaborativo se basa en un dataset\
                con 100.004 ratings de 671 usuarios.Los datasets están\
                disponible en:\
                https://www.kaggle.com/rounakbanik/the-movies-dataset.')
    

# If the option 'Recomendación general' is selected, execute this code
if tipo == 'Recomendación general':
    
    st.header('Recomendación general')
    
    st.write('')
    
    st.write('Elegí un género (opcional) e idioma original (opcional) y obtené\
             tu recomendación.')
    
    genre = st.selectbox(label='Género', options=genres)
    
    language = st.selectbox(label='Idioma', options=[None,'en','fr','ja','de',\
                                                   'it','es'])
    
    st.write('')

    if st.button('¡Recomendar!'):
        st.write('')
        st.table(general_recommendation(genre=genre, original_language=language))

# Create a selection of movies with more than 1000 votes.     
movies_select = df_small.title[df_small.vote_count>1000].sort_values().tolist()

# If the option 'Recomendación por contenido' is selected, execute this code
if tipo == 'Recomendación por contenido':
    
    st.header('Recomendación por contenido')
    
    st.write('')
    
    st.write('Elegí una película que te guste y obtené una recomendación de\
             peliculas similares a la película de entrada.')
    
    movie = st.selectbox(label='', options=movies_select) # only a selection
    # of movies is shown in the selection box to make the amount of movies
    # to choose from in the list manageable.
    
    st.write('')
    
    if st.button('¡Recomendar!'):
        st.write('')
        st.table(recommendation_content(movie))
  
# If the option 'Recomendación por filtro colaborativo' is selected, execute
# this code.        
if tipo == 'Recomendación por filtro colaborativo':
    
    st.header('Recomendación por filtro colaborativo')
    
    st.write('')
    
    st.write('Obtené una recomendación de 10 películas, comparando tus\
             preferencias con los ratings de otros usuarios.')
    
    movies_pos = st.multiselect(label='Elegí al menos 5 películas que te hayan\
                                gustado mucho',options=movies_select)
    movies_neg = st.multiselect(label='Elegí al menos 3 películas que NO te\
                                hayan gustado',options=movies_select)
    
    my_ratings=[]
    
    for movie in movies_pos:
        my_ratings.append({'userId': (max(ratings.userId)+1),\
                            'movieId':df_small.movieId[df_small.title ==\
                                                       movie].values[0],\
                            'rating':5})
        
    for movie in movies_neg:
        my_ratings.append({'userId': (max(ratings.userId)+1),\
                            'movieId':df_small.movieId[df_small.title ==\
                                                       movie].values[0],\
                            'rating':1})
            
    st.write('')
        
    if st.button('¡Recomendar!'):
        st.write('')
        st.table(recommendation_nmf(my_ratings)) 
