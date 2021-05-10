import pandas as pd
import numpy as np
import streamlit as st

import fuzzywuzzy
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer, models
from transformers import CamembertModel, CamembertTokenizer


# define french stopwords
stops = """alors au aucuns aussi autre avant avec avoir bon car ce cela ces ceux chaque ci comme comment 
    dans des du dedans dehors depuis devrait doit donc dos début elle elles en encore essai est et eu fait 
    faites fois font hors ici il ils je juste la le les leur là ma maintenant mais mes mien moins mon mot 
    même ni nommés notre nous ou où par parce pas peut peu plupart pour pourquoi quand que quel quelle quelles 
    quels qui sa sans ses seulement si sien son sont sous soyez sujet sur ta tandis tellement tels tes ton tous 
    tout trop très tu voient vont votre vous vu ça étaient état étions été être de à aux g mg""".split(" ")


def search_engine(df,columns,product,nlp_model, allergens, nutriscore):
    df['content'] = df[columns].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
    df['content'].fillna('Null', inplace = True)

    ## Vectorize and fit_transform to create matrix based on TFIDF method
    if nlp_model == 'TFIDF':
        tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = stops)
        tfidf_matrix = tf.fit_transform(df['content'])

        # create cosine similarities between input product name and all other products
        input_matrix = tf.transform([product])
        cosine_similarities = linear_kernel(input_matrix, tfidf_matrix)

        # print top 10 products with best similarity score 
        similar_indices = cosine_similarities[0].argsort()[:-10:-1]

        for i in similar_indices:
            if (df['allergens_0'][i] not in allergens) and (df['nutrition_grade'][i] in nutriscore):
                st.header('Produit : ' + df['product_name'][i])
                st.subheader('Marque : ' + df['brands'][i])
                st.write('**Ingrédients** : ' + df['ingredients'][i])

                # display two columns with infomations about product and informations about nutriments
                col1, col2 = st.beta_columns(2)
                df_informations = df.copy()
                df_nutriments = df.copy()

                col1.header("Informations diverses sur le produit")
                df_informations = df_informations[['labels', 'allergens_0', 'traces', 'additives_n', 'nutrition_grade']]
                col1.dataframe(df_informations.loc[i])

                col2.header("Informations Nutritionnelles")
                df_nutriments = df_nutriments[['energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 
                                            'proteins_100g', 'sodium_100g', 'alcohol_100g', 'fruits_vegetables_nuts', 'glycemic-index_100g']]
                col2.dataframe(df_nutriments.loc[i])
                st.markdown('___________')

            else:
                i+1

    
    ## Vectorize and fit_transform to create matrix based on CountVectorizer method
    elif nlp_model == 'CountVectorizer':
        tf = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = stops)
        tfidf_matrix = tf.fit_transform(df['content'])

        # create cosine similarities between input product name and all other products
        input_matrix = tf.transform([product])
        cosine_similarities = linear_kernel(input_matrix, tfidf_matrix)


        # print top 10 products with best similarity score 
        similar_indices = cosine_similarities[0].argsort()[:-10:-1]

        for i in similar_indices:
            if (df['allergens_0'][i] not in allergens) and (df['nutrition_grade'][i] in nutriscore):
                st.header('Produit : ' + df['product_name'][i])
                st.subheader('Marque : ' + df['brands'][i])
                st.write('**Ingrédients** : ' + df['ingredients'][i])

                # display two columns with infomations about product and informations about nutriments
                col1, col2 = st.beta_columns(2)
                df_informations = df.copy()
                df_nutriments = df.copy()

                col1.header("Informations diverses sur le produit")
                df_informations = df_informations[['labels', 'allergens_0', 'traces', 'additives_n', 'nutrition_grade']]
                col1.dataframe(df_informations.loc[i])

                col2.header("Informations Nutritionnelles")
                df_nutriments = df_nutriments[['energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 
                                            'proteins_100g', 'sodium_100g', 'alcohol_100g', 'fruits_vegetables_nuts', 'glycemic-index_100g']]
                col2.dataframe(df_nutriments.loc[i])
                st.markdown('___________')

            else:
                i+1

def replace_matches_in_column(df, column, string_to_match, min_ratio = 50):   
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # # only get matches with a ratio > 50
    return matches[0][0]