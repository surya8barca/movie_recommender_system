#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#setting up dataset


# In[7]:


column_list=['user_id','item_id','rating','timestamp']


# In[12]:


user_data=pd.read_csv('u.data',sep='\t',names=column_list)


# In[13]:


user_data.head()


# In[14]:


movie=pd.read_csv('Movie_Id_Titles')


# In[15]:


movie.head()


# In[17]:


new_data=pd.merge(user_data,movie,on='item_id')


# In[18]:


new_data.head()


# In[19]:


#data modification,analysis and visualization of dataset


# In[23]:


new_data.groupby('title')['rating'].mean().sort_values(ascending=False) #avg rating of movies


# In[25]:


new_data.groupby('title')['rating'].count().sort_values(ascending=False) #count of ratings


# In[26]:


ratings=pd.DataFrame(new_data.groupby('title')['rating'].mean())


# In[32]:


ratings.head()


# In[33]:


ratings['Number of Ratings']=new_data.groupby('title')['rating'].count()


# In[34]:


ratings.head()


# In[37]:


ratings['rating'].hist(bins=50)


# In[38]:


ratings['Number of Ratings'].hist(bins=50)


# In[39]:


sns.jointplot(x='rating',y='Number of Ratings',data=ratings)


# In[40]:


#creating recommender


# In[42]:


new_data.head()


# In[44]:


movie_mat=new_data.pivot_table(index='user_id',columns='title',values='rating')


# In[46]:


movie_mat


# In[49]:


ratings.sort_values('Number of Ratings',ascending=False).head()


# In[50]:


#Now we will create the recommender of top most rated movies i.e for Star Wars and Contact


# In[51]:


#Star Wars Recommender


# In[52]:


star_wars_ratings=movie_mat['Star Wars (1977)']


# In[56]:


star_wars_ratings.head()


# In[62]:


similar_to_star_wars=movie_mat.corrwith(star_wars_ratings)


# In[63]:


similar_to_star_wars.head()


# In[64]:


corr_starwars=pd.DataFrame(similar_to_star_wars,columns=['Correlation'])


# In[68]:


corr_starwars.head().sort_values(by='Correlation',ascending=False)


# In[69]:


corr_starwars = corr_starwars.join(ratings['Number of Ratings'])


# In[84]:


corr_starwars.head()


# In[ ]:


#Top 10 recommendations related to Star Wars


# In[86]:


corr_starwars[corr_starwars['Number of Ratings']>100].sort_values('Correlation',ascending=False).head(10)


# In[ ]:





# In[72]:


#Contact movie Recommender


# In[73]:


contact_ratings=movie_mat['Contact (1997)']


# In[75]:


contact_ratings.head()


# In[76]:


similar_to_contact=movie_mat.corrwith(contact_ratings)


# In[78]:


similar_to_contact.head()


# In[79]:


corr_contact=pd.DataFrame(similar_to_contact,columns=['Correlation'])


# In[80]:


corr_contact.head()


# In[81]:


corr_contact=corr_contact.join(ratings['Number of Ratings'])


# In[83]:


corr_contact.head()


# In[85]:


#Top 10 recommendations related to Contact


# In[87]:


corr_contact[corr_contact['Number of Ratings']>100].sort_values('Correlation',ascending=False).head(10)


# In[ ]:




