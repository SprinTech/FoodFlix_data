# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st


# %%
df = pd.read_csv('../Data/02_processed/products_prepared.csv')


# %%
df.head(5)

# %% [markdown]
# ## Exploratory Data Analysis

# %%
df['brands'] = df['brands'].astype('str')
df['product_name'] = df['product_name'].astype('str')


# %%
brands_corpus = ' '.join(df['brands'])
products_corpus = ' '.join(df['product_name'])


# %%
# define french stopwords
stops = "alors au aucuns aussi autre avant avec avoir bon car ce cela ces ceux chaque ci comme comment dans des du dedans dehors depuis devrait doit donc dos début elle elles en encore essai est et eu fait faites fois font hors ici il ils je juste la le les leur là ma maintenant mais mes mien moins mon mot même ni nommés notre nous ou où par parce pas peut peu plupart pour pourquoi quand que quel quelle quelles quels qui sa sans ses seulement si sien son sont sous soyez sujet sur ta tandis tellement tels tes ton tous tout trop très tu voient vont votre vous vu ça étaient état étions été être de à aux g mg".split(" ")


# %%
brands_wordcloud = WordCloud(stopwords = stops, background_color = 'white', collocations=False, height = 2000, width = 4000).generate(brands_corpus)
plt.figure(figsize = (16,8))
plt.imshow(brands_wordcloud)
plt.axis('off')
plt.show()


# %%
products_wordcloud = WordCloud(stopwords = stops, background_color = 'white', collocations=False, height = 2000, width = 4000).generate(products_corpus)
plt.figure(figsize = (16,8))
plt.imshow(products_wordcloud)
plt.axis('off')
plt.show()


# %%
df['content'] = df[['brands', 'product_name']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)


# %%
df['content'].fillna('Null', inplace = True)

# %% [markdown]
# ## Training recommander

# %%
tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = stops)
tfidf_matrix = tf.fit_transform(df['content'])


# %%
input_matrix = tf.transform("caramel")
cosine_similarities = linear_kernel(input, tfidf_matrix)


# %%
results = {}
for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()
    similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]

# %% [markdown]
# ## Prediction

# %%
def item(id):
    name   = df.loc[df['id'] == id]['content'].tolist()[0].split(' // ')[0]
    desc   = ' \nDescription: ' + df.loc[df['id'] == id]['content'].tolist()[0].split(' // ')[1][0:165] + '...'
    prediction = name  + desc
    return prediction

def recommend(item_id, num):
    print('Recommending ' + str(num) + ' products similar to ' + item(item_id))
    print('---')
    recs = results[item_id][:num]
    for rec in recs:
        print('\nRecommended: ' + item(rec[1]) + '\n(score:' + str(rec[0]) + ')')


# %%
recommend(item_id = 140, num = 5)


