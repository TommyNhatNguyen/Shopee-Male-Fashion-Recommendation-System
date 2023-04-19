import streamlit as st 
import numpy as np 
import pandas as pd 
from gensim import corpora, models, similarities
import content_based as cbm
from PIL import Image
import requests
from scipy import stats
from streamlit_image_select import image_select

# ---------- Model -----------
stop_words = cbm.stop_words
# read data for content based
@st.cache_data
def load_data():
    df = pd.read_parquet("./data/final_data/Products_ThoiTrangNam_streamlit_final.parquet", engine='pyarrow')
    tfidf = models.TfidfModel.load("./models/gensim_tfidf_model.tfidf")
    corpus = corpora.MmCorpus('./models/gensim_corpus.mm')
    dictionary = corpora.Dictionary.load('./models/gensim_dictionary.dict')
    index = similarities.SparseMatrixSimilarity.load('./models/gensim_index.index')
    result = [df, tfidf, corpus, dictionary, index]
    return result
df, tfidf, corpus, dictionary, index = load_data()

@st.cache_data
def load_historical_data():
    historical = pd.read_parquet("./data/final_data/historical_data.parquet", engine="pyarrow")
    return historical
historical_df = load_historical_data()

@st.cache_data
def load_als_model():
    als_recommendation = pd.read_parquet('./data/final_data/recommend_results_collaborative.parquet', engine='pyarrow')
    return als_recommendation
als_recommendation = load_als_model()

def show_product(search_results):
    num_product_per_row = 5
    i = 0
    for row in np.arange(num_product_show/num_product_per_row):
        columns = st.columns(5)
        for col in columns:
            try:
                product_con = st.container()
                with product_con:
                # get each product's information
                    product_image = Image.open(requests.get(search_results['image'].iloc[i], stream=True).raw)
                    product_link = search_results['link'].iloc[i]
                    product_id = search_results['product_id'].iloc[i]
                    product_name = search_results['product_name'].iloc[i]
                    product_price = search_results['price'].iloc[i]
                    product_rating = search_results['rating'].iloc[i]
                    product_category = search_results['sub_category'].iloc[i]
                    if search_results['image'].iloc[i] is not None:
                        product_image = Image.open(requests.get(search_results['image'].iloc[i], stream=True).raw)
                        col.image(product_image)
                    else:
                        col.image("./images/no_image.jpg")
                    col.write(f"[{str(product_name).capitalize()}]({product_link})")
                    col.write(f"Price: {product_price:,.0f} vnđ")
                    col.write(f"Rating: {product_rating}/5")
                    i += 1
            except:
                continue


# ----------- GUI ------------- 
# Add Sidebar
content_menu = ("Application Overview", "Start Recommendation")
content_select = st.sidebar.selectbox(
    "Application Content",
    content_menu
)
# ---------
# TODO1: Application Overview
if content_select == content_menu[0]:
    st.write("# Male Fashion Recommendation on Shopee")
    st.image("./images/shopee.jpg")
    st.divider()

    # 1. Business Objectives 
    # Add header
    st.write("## **I. Business Understanding:**")
    # Add content and image
    col1, col2 = st.columns(2)
    with col1:
        st.write("""**Recommend relevant products to customers is a win-win scenario for both customers and e-commerce retailers.**
        It helps users to get a better experience by helping customers find products they are looking for, personalize by buying experience.
        More importantly, this helps business to increase their revenue by offering their customers to browse and purchase more relevant products, hence increase sales and customer lifetime value.""")
    with col2:
        st.image("./images/money.jpg")
    # 2. Methods
    st.write("## **II. Methods:**")
    # Add content for Content-Based Filtering
    st.write("### **1. Content-Based Filtering:**")
    st.write("""**Content-based Filtering uses item features to recommend other items similar to 
    what the user likes, based on their previous actions or explicit feedback.**
    The recommender system will recommend relevant products to the product that the user has searched.
    This helps business to increase sales by getting customers to buy more relevant products.""")
    st.image("./images/content_based.jpg")
    st.write("""**Model used for content-based filtering is Gensim TF-IDF model.** 
    Gensim is a free open-source Python library for representing documents as semantic vectors, it is used to process plain text using unsupervised machine learning algorithms.
    TF-IDF (Term Frequency - Inverse Document Frequency) is an algorithm that uses the frequency of words to determine how relevant those words are to a given document.""")
    st.image("./images/gensim.jpg")
    # Add content for Collaborative Filtering
    st.write("### **2. Collaborative Filtering:**")
    st.write("""**Collaborative Filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.** 
    It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user.
    Similarly to Content-Based Recommendation, Collaborative Filtering also help business to increase their sales by offering products that might fit a potential customer based on other customer's preferences.""")
    st.write("""**Model used for collaborative filtering is ALS model (Alternating Least Square)**. """)
    st.image("./images/colab.png")
    # 3. Recommendation Flow 
    st.write("## **III. Recommendation Flow Chart:**")
    st.image("./images/flow_chart.png")
# ---------
elif content_select == content_menu[1]:
    st.write("# Male Fashion Recommendation on Shopee")
    st.image("./images/shopee.png")
    log_in_check = ("Content-Based Filtering", "Collaborative Filtering")
    have_account = st.radio("Select a recommendation method:", log_in_check)
    # Trường hợp 1: Content-Based Filtering ----------------
    if have_account == log_in_check[0]:
        st.divider()
        st.write("## Content-Based Filtering")
        st.image("./images/content_based.jpg")
        st.write("### Let's search your favorite product on Shopee 👇")
        # Add search box
        user_search = st.text_input("Search Male Fashion:", placeholder="Type a product you need", key='user_search', value="")
        num_product_show = st.select_slider('Select number of recommend products:', [5, 10, 15, 20, 30, 50, 100], value=5, key='num_product_show3')
        if user_search != "":
            st.write("## Search results 👇")
            # Get recommend result for user's input
            search_results = cbm.search_similar_product(df, index, tfidf, dictionary, user_search)
            image_links = search_results['image'].fillna("./images/no_image.jpg").tolist()[:num_product_show]
            product_info = []
            # Get each product's information
            for i in range(num_product_show):
                product_name = search_results['product_name'].iloc[i]
                product_price = search_results['price'].iloc[i]
                product_rating = search_results['rating'].iloc[i]
                caption = f"""{product_name} - Price: {product_price:,.0f} vnđ - Rating: {product_rating}/5"""
                product_info.append(caption)
            # Show recommendation
            imgs = image_select(f'Top {num_product_show} Relevant Products', image_links, product_info, use_container_width=False, return_value='index')
            # -----------Search for select image------------ 
            st.divider()
            st.write("## Recommendation from selected product 👇")
            select_product = search_results.iloc[imgs, :]['product_name_description_wt']
            select_results = cbm.search_similar_product(df, index, tfidf, dictionary, select_product)
            # Show on GUI
            i = 0
            num_product_per_row=5
            for row in np.arange(num_product_show/num_product_per_row):
                columns = st.columns(5)
                for col in columns:
                    product_con = st.container()
                    with product_con:
                        # get each product's information
                        product_link = select_results['link'].iloc[i]
                        product_id = select_results['product_id'].iloc[i]
                        product_name = select_results['product_name'].iloc[i]
                        product_price = select_results['price'].iloc[i]
                        product_rating = select_results['rating'].iloc[i]
                        product_category = select_results['sub_category'].iloc[i]
                        if select_results['image'].iloc[i] is not None:
                            product_image = Image.open(requests.get(select_results['image'].iloc[i], stream=True).raw)
                            col.image(product_image)
                        else:
                            col.image("./images/no_image.jpg")
                        col.write(f"[{str(product_name).capitalize()}]({product_link})")
                        col.write(f"Price: {product_price:,.0f} vnđ")
                        col.write(f"Rating: {product_rating}/5")
                        i += 1

    # Trường hợp 2: Collaborative Filtering ----------------
    else: 
        st.divider()
        st.write("## Collaborative Filtering")
        st.image("./images/colab.png")
        list_of_users = historical_df['user'].unique()
        login_account = st.selectbox("Select a username:", list_of_users, index=3)
        num_product_show = st.select_slider('Select number of recommend products:', [5, 10, 15, 20, 30, 50, 100], value=5, key='num_product_show1')
        st.write(f"## Shopee found products {login_account} might like 👇")
        # Get chosen user_id in als_recommendation
        login_account_als_recommendation = als_recommendation[als_recommendation['user'] == login_account]
        # Get recommend result for selected user_id 
        image_links = login_account_als_recommendation['image'].fillna("./images/no_image.jpg").tolist()[:num_product_show]
        product_info = []
        # get each product's information
        for i in range(num_product_show):
            try:
                product_name = login_account_als_recommendation['product_name'].iloc[i]
                product_price = login_account_als_recommendation['price'].iloc[i]
                product_rating = login_account_als_recommendation['rating'].iloc[i]
                caption = f"""{product_name} - Price: {product_price} vnđ - Rating: {product_rating}/5"""
                product_info.append(caption)
            except:
                continue
        # Show GUI
        imgs = image_select(f'Top {num_product_show} Relevant Products', image_links, product_info, use_container_width=False, return_value='index')
        # -----------Search for select image------------ 
        st.divider()
        st.write("## Recommendation from selected product 👇")
        select_product = login_account_als_recommendation.iloc[imgs, :]['product_name_description_wt']
        select_results = cbm.search_similar_product(df, index, tfidf, dictionary, select_product)
        # Show on GUI
        i = 0
        num_product_per_row=5
        for row in np.arange(num_product_show/num_product_per_row):
            columns = st.columns(5)
            for col in columns:
                product_con = st.container()
                with product_con:
                    # get each product's information
                    product_link = select_results['link'].iloc[i]
                    product_id = select_results['product_id'].iloc[i]
                    product_name = select_results['product_name'].iloc[i]
                    product_price = select_results['price'].iloc[i]
                    product_rating = select_results['rating'].iloc[i]
                    product_category = select_results['sub_category'].iloc[i]
                    if select_results['image'].iloc[i] is not None:
                        product_image = Image.open(requests.get(select_results['image'].iloc[i], stream=True).raw)
                        col.image(product_image)
                    else:
                        col.image("./images/no_image.jpg")
                    col.write(f"[{str(product_name).capitalize()}]({product_link})")
                    col.write(f"Price: {product_price:,.0f} vnđ")
                    col.write(f"Rating: {product_rating}/5")
                    i += 1

# --------- Exception ---------
else:
    st.write("# Something went wrong! Please contact developers for more info.")