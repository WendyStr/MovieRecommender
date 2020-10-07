## Movie Recommender System

### Purpose of project

This movie recommendation system was created as an integrating project for the Data Science course in Digital House. The purpose of this project was to put the different skills acquired during the course into practice.

### Introduction on recommender systems

Today's world is full of options and it is often difficult for us to know what we want. The goal of a recommender system is to recommend items to a user that are likely to be of interest to them. A recommender system aims to predict users' interests and acts as an information filtering system. 

We can broadly distinguish 4 classes of recommender systems:
- **General systems:** based on popularity or ratings.
- **Systems based on content:** based on descriptions of the items and profiles of users preferences.
- **Systems based on collaborative filtering:** recommends items solely based on the ratings of other users. It assumes that if two users have a similar opinion on an item, they will be more likely to have a similar opinion on a different item as well.
- **Hybrid systems:** a system that combines the above classes.

I will focus on the first three classes in this project (general, content and collaborative filtering). 

### Dataset

The dataset used for this project is 'The Movies Dataset' from Kaggle, and can be found under the following link: https://www.kaggle.com/rounakbanik/the-movies-dataset

I have used the following files:

- **movies_metadata.csv**: Main file with movie metadata. Contains information on 45,466 movies presented in the Full MovieLens dataset. Features include budgets, revenues, release dates, languages, genres, production countries, and production companies.
- **keywords.csv**: Contains the keywords of the movie plots.
- **credits.csv**: Contains information about the cast and crew of each movie. 
- **links_small.csv**: Contains the TMDB and IMDB id's of a subset of 9,066 movies from the full data set.
- **ratings_small.csv**: A subset of 100,004 ratings from 671 users on 9,066 movies.

The original dataset contains information about 45,466 movies. However, due to computational limitations, I had to work with a reduced dataset. After reducing and cleaning the dataset, I had 9010 movies left to use in the recommender system.

### Libraries

pandas: 1.0.3
<br>
numpy: 1.18.1
<br>
matplotlib: 3.1.3
<br>
seaborn: 0.10.1
<br>
pickleshare: 0.7.5
<br>
scikit-learn: 0.22.1
<br>
scikit-surprise: 1.1.0
<br>
streamlit: 0.62.1