import streamlit as st
import numpy as np
import pandas as pd
import fuzzywuzzy
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from functions import *

df = pd.read_csv('Data/02_processed/products_prepared.csv')

st.title('Moteur de recommendation FoodFlix')

# Get user input
product = st.sidebar.text_input('Quel produit recherchez vous ?')

s = df["allergens_0"].value_counts()
allergens = st.sidebar.multiselect('Quels allergènes souhaitez-vous exclure ?', s[s > 100].index)

nutriscore = st.sidebar.multiselect('Quel nutriscore souhaitez-vous pour votre produit ?', df['nutrition_grade'].unique())

nlp_model = st.sidebar.radio('Type de recherche souhaitée : ' , ['TFIDF', 'CountVectorizer'])

# correct typo from user
if product:
    product = replace_matches_in_column(df, 'product_name', product)
    st.write('Produit de référence : **{}**'.format(product))

# give recommendations based of user research
    search_engine(df, ['brands', 'product_name'], product, nlp_model, allergens, nutriscore)
