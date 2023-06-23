# Shopee Male Fashion Recommendation System 
![image](https://github.com/TommyNhatNguyen/shopee_recommendation/assets/86128966/a2d111ec-7d04-43ec-9ca5-f37ca0a04682)

**My application link:** [Click here](https://tommynhatnguyen-shopee-recommendation-project-final-ikps3f.streamlit.app/)

**Source code:** ![My code file](https://github.com/TommyNhatNguyen/shopee_recommendation/blob/main/project_final.py)
### 1. Business Understanding: 
- **Recommend relevant products to customers is a win-win scenario for both customers and e-commerce retailers**. Customers can easily find products they need in a short-time, and business can increase their revenue by offering relevant products for each customers.
### 2. Methods: 
#### 2.1 Content-Based Filtering: 
![image](https://github.com/TommyNhatNguyen/shopee_recommendation/assets/86128966/b69f35e9-5b90-4d2f-b00f-b0404448364d)
- Content-based Filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.
- Model: Gensim TF-IDF model
#### 2.2 Collaborative Filtering:
![image](https://github.com/TommyNhatNguyen/shopee_recommendation/assets/86128966/ad25ef10-6e39-4b96-a174-efaf853e6aa0)
- Collaborative Filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.
- Model: ALS model (Alternating Least Square)
### 3. Application Overview:

- **Step 1:** Select a recommendation method
![image](https://github.com/TommyNhatNguyen/shopee_recommendation/assets/86128966/220e143e-5fef-47c2-ab8d-9e342dee15e7)

- **Step 2:** Input product name, select number of recommend product
![image](https://github.com/TommyNhatNguyen/shopee_recommendation/assets/86128966/ee150ecb-a414-48e8-8a9f-6f5640730291)

- **Step 3:** Click on recommend results to show more recommendation
![image](https://github.com/TommyNhatNguyen/shopee_recommendation/assets/86128966/65365ca1-770d-4025-a84b-14bc3127607b)

