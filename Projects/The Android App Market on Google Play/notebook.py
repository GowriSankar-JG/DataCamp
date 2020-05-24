#!/usr/bin/env python
# coding: utf-8

# ## 1. Google Play Store apps and reviews
# <p>Mobile apps are everywhere. They are easy to create and can be lucrative. Because of these two factors, more and more apps are being developed. In this notebook, we will do a comprehensive analysis of the Android app market by comparing over ten thousand apps in Google Play across different categories. We'll look for insights in the data to devise strategies to drive growth and retention.</p>
# <p><img src="https://assets.datacamp.com/production/project_619/img/google_play_store.png" alt="Google Play logo"></p>
# <p>Let's take a look at the data, which consists of two files:</p>
# <ul>
# <li><code>apps.csv</code>: contains all the details of the applications on Google Play. There are 13 features that describe a given app.</li>
# <li><code>user_reviews.csv</code>: contains 100 reviews for each app, <a href="https://www.androidpolice.com/2019/01/21/google-play-stores-redesigned-ratings-and-reviews-section-lets-you-easily-filter-by-star-rating/">most helpful first</a>. The text in each review has been pre-processed and attributed with three new features: Sentiment (Positive, Negative or Neutral), Sentiment Polarity and Sentiment Subjectivity.</li>
# </ul>

# In[134]:


# Read in dataset
import pandas as pd
apps_with_duplicates = pd.read_csv('datasets/apps.csv')

# Drop duplicates
apps = apps_with_duplicates.drop_duplicates()

# Print the total number of apps
print('Total number of apps in the dataset = ', len(apps))

# Have a look at a random sample of 5 rows
n = 5
apps.sample(n)


# In[135]:


get_ipython().run_cell_magic('nose', '', '\ncorrect_apps_with_duplicates = pd.read_csv(\'datasets/apps.csv\')\n\ndef test_pandas_loaded():\n    assert (\'pd\' in globals()), "pandas is not imported and aliased as specified in the instructions."\n\ndef test_apps_with_duplicates_loaded():\n#     correct_apps_with_duplicates = pd.read_csv(\'datasets/apps.csv\')\n    assert (correct_apps_with_duplicates.equals(apps_with_duplicates)), "The data was not correctly read into apps_with_duplicates."\n    \ndef test_duplicates_dropped():\n#     correct_apps_with_duplicates = pd.read_csv(\'datasets/apps.csv\')\n    correct_apps = correct_apps_with_duplicates.drop_duplicates()\n    assert (correct_apps.equals(apps)), "The duplicates were not correctly dropped from apps_with_duplicates."\n    \ndef test_total_apps():\n    correct_total_apps = len(correct_apps_with_duplicates.drop_duplicates())\n    assert (correct_total_apps == len(apps)), "The total number of apps is incorrect. It should equal 9659."\n    \ndef test_n():\n    correct_n = 5\n    assert (correct_n == n), "The number of records displayed is wrong. Select a random sample of 5 rows." ')


# ## 2. Data cleaning
# <p>The three features that we will be working with most frequently henceforth are <code>Installs</code>, <code>Size</code>, and <code>Price</code>. A careful glance of the dataset reveals that some of these columns mandate data cleaning in order to be consumed by code we'll write later. Specifically, the presence of special characters (<code>, $ +</code>) and letters (<code>M k</code>) in the <code>Installs</code>, <code>Size</code>, and <code>Price</code> columns make their conversion to a numerical data type difficult. Let's clean by removing these and converting each column to a numeric type.</p>

# In[136]:


# List of characters to remove
chars_to_remove = ['+',',','M','$']
# List of column names to clean
cols_to_clean = ['Installs','Size','Price']

# Loop for each column
for col in cols_to_clean:
    # Replace each character with an empty string
    for char in chars_to_remove:
        apps[col] = apps[col].str.replace(char, '')
    # Convert col to numeric
    apps[col] = pd.to_numeric(apps[col]) 


# In[137]:


get_ipython().run_cell_magic('nose', '', '\ndef test_installs_plus():\n    assert \'+\' not in apps[\'Installs\'], \\\n    \'Some of the "+" characters still remain in the Installs column.\' \n    \ndef test_installs_comma():\n    assert \',\' not in apps[\'Installs\'], \\\n    \'Some of the "," characters still remain in the Installs column.\'\n    \ndef test_installs_numeric():\n    assert type(apps[\'Installs\'][0]) != \'numpy.int64\', \\\n    \'The Installs column is not of numeric data type (int).\'\n    \ndef test_size_M():\n    assert \'M\' not in apps[\'Size\'], \\\n    \'Some of the "M" characters still remain in the Size column.\'\n    \ndef test_size_comma():\n    assert \',\' not in apps[\'Size\'], \\\n    \'Some of the "," characters still remain in the Size column.\'\n\ndef test_size_k():\n    assert \'k\' not in apps[\'Size\'], \\\n    \'Some of the "k" characters still remain in the Size column.\'\n    \ndef test_size_numeric():\n    assert type(apps[\'Size\'][0]) != \'numpy.float64\', \\\n    \'The Size column is not of numeric data type (float).\'\n    \ndef test_price_dollar():\n    assert \'$\' not in apps[\'Price\'], \\\n    \'Some of the "$" characters still remain in the Price column.\'\n\ndef test_price_numeric():\n    assert type(apps[\'Price\'][0]) != \'numpy.float64\', \\\n    \'The Price column is not of numeric data type (float).\'')


# ## 3. Exploring app categories
# <p>With more than 1 billion active users in 190 countries around the world, Google Play continues to be an important distribution platform to build a global audience. For businesses to get their apps in front of users, it's important to make them more quickly and easily discoverable on Google Play. To improve the overall search experience, Google has introduced the concept of grouping apps into categories.</p>
# <p>This brings us to the following questions:</p>
# <ul>
# <li>Which category has the highest share of (active) apps in the market? </li>
# <li>Is any specific category dominating the market?</li>
# <li>Which categories have the fewest number of apps?</li>
# </ul>
# <p>We will see that there are <code>33</code> unique app categories present in our dataset. <em>Family</em> and <em>Game</em> apps have the highest market prevalence. Interestingly, <em>Tools</em>, <em>Business</em> and <em>Medical</em> apps are also at the top.</p>

# In[138]:


import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Print the total number of unique categories
num_categories = len(apps['Category'].unique())
print('Number of categories = ', num_categories)

# Count the number of apps in each 'Category' and sort them in descending order
num_apps_in_category = apps['Category'].value_counts().sort_values(ascending = False)

data = [go.Bar(
        x = num_apps_in_category.index, # index = category name
        y = num_apps_in_category.values, # value = count
)]

plotly.offline.iplot(data)


# In[139]:


get_ipython().run_cell_magic('nose', '', '\n# last_value = _\n# print(type(last_value))\n\ndef test_num_categories():\n    assert num_categories == 33, "The number of app categories is incorrect. It should equal 33."\n    \ndef test_num_apps_in_category():\n    correct_num_apps_in_category = apps[\'Category\'].value_counts().sort_values(ascending=False)\n    assert (correct_num_apps_in_category == num_apps_in_category).all(), "num_apps_in_category is not what we expected. Please inspect the hint."')


# ## 4. Distribution of app ratings
# <p>After having witnessed the market share for each category of apps, let's see how all these apps perform on an average. App ratings (on a scale of 1 to 5) impact the discoverability, conversion of apps as well as the company's overall brand image. Ratings are a key performance indicator of an app.</p>
# <p>From our research, we found that the average volume of ratings across all app categories is <code>4.17</code>. The histogram plot is skewed to the right indicating that the majority of the apps are highly rated with only a few exceptions in the low-rated apps.</p>

# In[140]:


# Average rating of apps
avg_app_rating = apps['Rating'].mean()
print('Average app rating = ', avg_app_rating)

# Distribution of apps according to their ratings
data = [go.Histogram(
        x = apps['Rating']
)]

# Vertical dashed line to indicate the average app rating
layout = {'shapes': [{
              'type' :'line',
              'x0': avg_app_rating,
              'y0': 0,
              'x1': avg_app_rating,
              'y1': 1000,
              'line': { 'dash': 'dashdot'}
          }]
          }

plotly.offline.iplot({'data': data, 'layout': layout})


# In[141]:


get_ipython().run_cell_magic('nose', '', '\ndef test_app_avg_rating():\n    assert round(avg_app_rating, 5) == 4.17324, \\\n    "The average app rating rounded to five digits should be 4.17324."\n    \n# def test_x_histogram():\n#     correct_x_histogram = apps[\'Rating\']\n#     assert correct_x_histogram.all() == data[0][\'x\'].all(), \\\n#     \'x should equal Rating column\'')


# ## 5. Size and price of an app
# <p>Let's now examine app size and app price. For size, if the mobile app is too large, it may be difficult and/or expensive for users to download. Lengthy download times could turn users off before they even experience your mobile app. Plus, each user's device has a finite amount of disk space. For price, some users expect their apps to be free or inexpensive. These problems compound if the developing world is part of your target market; especially due to internet speeds, earning power and exchange rates.</p>
# <p>How can we effectively come up with strategies to size and price our app?</p>
# <ul>
# <li>Does the size of an app affect its rating? </li>
# <li>Do users really care about system-heavy apps or do they prefer light-weighted apps? </li>
# <li>Does the price of an app affect its rating? </li>
# <li>Do users always prefer free apps over paid apps?</li>
# </ul>
# <p>We find that the majority of top rated apps (rating over 4) range from 2 MB to 20 MB. We also find that the vast majority of apps price themselves under \$10.</p>

# In[142]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")

# Subset for categories with at least 250 apps
large_categories = apps.groupby('Category').filter(lambda x: len(x) >= 250).reset_index()

# Plot size vs. rating
plt1 = sns.jointplot(y = large_categories['Rating'], x = large_categories['Size'], kind = 'hex')

# Subset out apps whose type is 'Paid'
paid_apps = apps[apps['Type'] == 'Paid']

# Plot price vs. rating
plt2 = sns.jointplot(y = paid_apps['Rating'], x = paid_apps['Price'])


# In[143]:


get_ipython().run_cell_magic('nose', '', '\ndef test_large_categories():\n    correct_large_categories = apps.groupby(\'Category\').filter(lambda x: len(x) >= 250).reset_index()\n    assert correct_large_categories.equals(large_categories), \\\n    "The large_categories DataFrame is not what we expected. Please review the instructions and check the hint if necessary."\n\ndef test_size_vs_rating():\n    correct_large_categories = apps.groupby(\'Category\').filter(lambda x: len(x) >= 250).reset_index()\n    correct_large_categories = correct_large_categories[correct_large_categories[\'Size\'].notnull()]\n    correct_large_categories = correct_large_categories[correct_large_categories[\'Rating\'].notnull()]\n    assert plt1.x.tolist() == correct_large_categories[\'Size\'].values.tolist() and plt1.y.tolist() == correct_large_categories[\'Rating\'].values.tolist(), \\\n    "The size vs. rating jointplot is not what we expected. Please review the instructions and check the hint if necessary."\n    \ndef test_paid_apps():\n    correct_paid_apps = apps[apps[\'Type\'] == \'Paid\']\n    assert correct_paid_apps.equals(paid_apps), \\\n    "The paid_apps DataFrame is not what we expected. Please review the instructions and check the hint if necessary."\n    \ndef test_price_vs_rating():\n    correct_paid_apps = apps[apps[\'Type\'] == \'Paid\']\n    correct_paid_apps = correct_paid_apps[correct_paid_apps[\'Price\'].notnull()]\n    correct_paid_apps = correct_paid_apps[correct_paid_apps[\'Rating\'].notnull()]\n    assert plt2.x.tolist() == correct_paid_apps[\'Price\'].values.tolist() and plt2.y.tolist() == correct_paid_apps[\'Rating\'].values.tolist(), \\\n    "The price vs. rating jointplot is not what we expected. Please review the instructions and check the hint if necessary."\n    ')


# ## 6. Relation between app category and app price
# <p>So now comes the hard part. How are companies and developers supposed to make ends meet? What monetization strategies can companies use to maximize profit? The costs of apps are largely based on features, complexity, and platform.</p>
# <p>There are many factors to consider when selecting the right pricing strategy for your mobile app. It is important to consider the willingness of your customer to pay for your app. A wrong price could break the deal before the download even happens. Potential customers could be turned off by what they perceive to be a shocking cost, or they might delete an app they’ve downloaded after receiving too many ads or simply not getting their money's worth.</p>
# <p>Different categories demand different price ranges. Some apps that are simple and used daily, like the calculator app, should probably be kept free. However, it would make sense to charge for a highly-specialized medical app that diagnoses diabetic patients. Below, we see that <em>Medical and Family</em> apps are the most expensive. Some medical apps extend even up to \$80! All game apps are reasonably priced below \$20.</p>

# In[144]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Select a few popular app categories
popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY',
                                            'MEDICAL', 'TOOLS', 'FINANCE',
                                            'LIFESTYLE','BUSINESS'])]

# Examine the price trend by plotting Price vs Category
ax = sns.stripplot(x = popular_app_cats['Price'], y = popular_app_cats['Category'], jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories')

# Apps whose Price is greater than 200
apps_above_200 = popular_app_cats[['Category', 'App', 'Price']][popular_app_cats['Price'] > 200]
apps_above_200


# In[145]:


get_ipython().run_cell_magic('nose', '', '\nlast_output = _\n\ndef test_apps_above_200():\n    assert len(apps_above_200) == 17, "There should be 17 apps priced above 200 in apps_above_200."')


# ## 7. Filter out "junk" apps
# <p>It looks like a bunch of the really expensive apps are "junk" apps. That is, apps that don't really have a purpose. Some app developer may create an app called <em>I Am Rich Premium</em> or <em>most expensive app (H)</em> just for a joke or to test their app development skills. Some developers even do this with malicious intent and try to make money by hoping people accidentally click purchase on their app in the store.</p>
# <p>Let's filter out these junk apps and re-do our visualization. The distribution of apps under \$20 becomes clearer.</p>

# In[146]:


# Select apps priced below $100
apps_under_100 = popular_app_cats[popular_app_cats['Price'] <100]

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Examine price vs category with the authentic apps
ax = sns.stripplot(x='Price', y='Category', data=apps_under_100,
                   jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories after filtering for junk apps')


# In[147]:


get_ipython().run_cell_magic('nose', '', '\ndef test_apps_under_100():\n    correct_apps_under_100 = popular_app_cats[popular_app_cats[\'Price\'] < 100]\n    assert correct_apps_under_100.equals(apps_under_100), \\\n    "The apps_under_100 DataFrame is not what we expected. Please review the instructions and check the hint if necessary."')


# ## 8. Popularity of paid apps vs free apps
# <p>For apps in the Play Store today, there are five types of pricing strategies: free, freemium, paid, paymium, and subscription. Let's focus on free and paid apps only. Some characteristics of free apps are:</p>
# <ul>
# <li>Free to download.</li>
# <li>Main source of income often comes from advertisements.</li>
# <li>Often created by companies that have other products and the app serves as an extension of those products.</li>
# <li>Can serve as a tool for customer retention, communication, and customer service.</li>
# </ul>
# <p>Some characteristics of paid apps are:</p>
# <ul>
# <li>Users are asked to pay once for the app to download and use it.</li>
# <li>The user can't really get a feel for the app before buying it.</li>
# </ul>
# <p>Are paid apps installed as much as free apps? It turns out that paid apps have a relatively lower number of installs than free apps, though the difference is not as stark as I would have expected!</p>

# In[148]:


trace0 = go.Box(
    # Data for paid apps
    y=apps[apps['Type'] == 'Paid']['Installs'],
    name = 'Paid'
)

trace1 = go.Box(
    # Data for free apps
    y=apps[apps['Type'] == 'Free']['Installs'],
    name = 'Free'
)

layout = go.Layout(
    title = "Number of downloads of paid apps vs. free apps",
    yaxis = dict(
        type = 'log',
        autorange = True
    )
)

# Add trace0 and trace1 to a list for plotting
data = [trace0,trace1]
plotly.offline.iplot({'data': data, 'layout': layout})


# In[149]:


get_ipython().run_cell_magic('nose', '', '\ndef test_trace0_y():\n    correct_y = apps[\'Installs\'][apps[\'Type\'] == \'Paid\']\n    assert all(trace0[\'y\'] == correct_y.values), \\\n    "The y data for trace0 appears incorrect. Please review the instructions and check the hint if necessary."\n\ndef test_trace1_y():\n    correct_y_1 = apps[\'Installs\'][apps[\'Type\'] == \'Free\']\n    correct_y_2 = apps[\'Installs\'][apps[\'Price\'] == 0]\n    try:\n        check_1 = all(trace1[\'y\'] == correct_y_1.values)\n    except:\n        check_1 = False\n    try:\n        check_2 = all(trace1[\'y\'] == correct_y_2.values)\n    except:\n        check_2 = False\n        \n    assert check_1 or check_2, \\\n    "The y data for trace1 appears incorrect. Please review the instructions and check the hint if necessary."')


# ## 9. Sentiment analysis of user reviews
# <p>Mining user review data to determine how people feel about your product, brand, or service can be done using a technique called sentiment analysis. User reviews for apps can be analyzed to identify if the mood is positive, negative or neutral about that app. For example, positive words in an app review might include words such as 'amazing', 'friendly', 'good', 'great', and 'love'. Negative words might be words like 'malware', 'hate', 'problem', 'refund', and 'incompetent'.</p>
# <p>By plotting sentiment polarity scores of user reviews for paid and free apps, we observe that free apps receive a lot of harsh comments, as indicated by the outliers on the negative y-axis. Reviews for paid apps appear never to be extremely negative. This may indicate something about app quality, i.e., paid apps being of higher quality than free apps on average. The median polarity score for paid apps is a little higher than free apps, thereby syncing with our previous observation.</p>
# <p>In this notebook, we analyzed over ten thousand apps from the Google Play Store. We can use our findings to inform our decisions should we ever wish to create an app ourselves.</p>

# In[150]:


# Load user_reviews.csv
reviews_df = pd.read_csv('datasets/user_reviews.csv')

# Join and merge the two dataframe
merged_df = pd.merge(apps, reviews_df, on = 'App', how = "inner")

# Drop NA values from Sentiment and Translated_Review columns
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)

# User review sentiment polarity for paid vs. free apps
ax = sns.boxplot(x = 'Type', y = 'Sentiment_Polarity', data = merged_df)
ax.set_title('Sentiment Polarity Distribution')


# In[151]:


get_ipython().run_cell_magic('nose', '', '\ndef test_user_reviews_loaded():\n    correct_user_reviews = pd.read_csv(\'datasets/user_reviews.csv\')\n    assert (correct_user_reviews.equals(reviews_df)), "The user_reviews.csv file was not correctly loaded. Please review the instructions and inspect the hint if necessary."\n    \ndef test_user_reviews_merged():\n    user_reviews = pd.read_csv(\'datasets/user_reviews.csv\')\n    correct_merged = pd.merge(apps, user_reviews, on = "App", how = "inner")\n    correct_merged = correct_merged.dropna(subset=[\'Sentiment\', \'Translated_Review\'])\n    assert (correct_merged.equals(merged_df)), "The merging of user_reviews and apps is incorrect. Please review the instructions and inspect the hint if necessary."')

