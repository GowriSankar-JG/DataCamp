#!/usr/bin/env python
# coding: utf-8

# # Analyzing Police Activity with pandas

# ## Course Description

# ### This course explores the Stanford Open Policing Project dataset and analyzes the impact of weather, time of day, reason for traffic stop and gender of driver on police behavior. It covers cleaning messy data, creating visualizations, combining and reshaping datasets, and manipulating time series data.

# ### Preparing data for analysis

# #### Before beginning the analysis, it is critical to first examine and clean the dataset, to make working with it a more efficient process. This chapter covers fixing data types, handling missing values, and dropping columns and rows while learning about the Stanford Open Policing Project dataset. 

# The dataset contains traffic stops by police officers, collected by the Stanford Open Policing Project. It has data on 31 US states. This course focuses on data from the state of Rhode Island. The full data can be downloaded from the project's website at https://openpolicing.stanford.edu/

# In[1]:


from IPython.display import Image
Image(filename='data/states.png')


# Before beginning your analysis, it's important that we familiarize ourselves with the dataset. 

# In[2]:


# Import the pandas library as pd
import pandas as pd

# Read 'police.csv' into a DataFrame named ri
ri = pd.read_csv('data/police.csv')

# Examine the head of the DataFrame
print(ri.head())

# Count the number of missing values in each column
print(ri.isnull().sum())


# It looks like most of the columns have at least some missing values.

# Often, a DataFrame will contain columns that are not useful to your analysis. Such columns should be dropped from the DataFrame, to make it easier for you to focus on the remaining columns. In this case, the state and the county_name columns are non-relevant, since we only focus on data from Rhode Island.

# In[3]:


# Examine the shape of the DataFrame
print(ri.shape)

# Drop the 'county_name' and 'state' columns
ri.drop(['county_name', 'state'], axis='columns', inplace=True)

# Examine the shape of the DataFrame (again)
print(ri.shape)


# When you know that a specific column will be critical to your analysis, and only a small fraction of rows are missing a value in that column, it often makes sense to remove those rows from the dataset.
# During this course, the `driver_gender` column will be critical to many of our analyses. Because only a small fraction of rows are missing `driver_gender`, we'll drop those rows from the dataset.

# In[4]:


# Count all observations with non-missing and missing 'driver_gender'
print(ri.driver_gender.count())
print(ri.driver_gender.isnull().sum())


# In[5]:


# Drop all rows that are missing 'driver_gender'
ri.dropna(subset=['driver_gender'], inplace=True)

# Count the number of missing values in each column (again)
print(ri.isnull().sum())

# Examine the shape of the DataFrame
print(ri.shape)


# We dropped around 5,000 rows, which is a small fraction of the dataset, and now only one column remains with any missing values.

# The data types of features were automatically inferred by `pandas` when reading in the *.csv* file. The data types currently in use are only `object` and `bool`. As data types affect which operations we can perform on a given Series, we should examine and fix data types in our dataset.

# In[6]:


ri.dtypes


# The `is_arrested` column currently has the object data type. We'll change the data type to bool, which is the most suitable type for a column containing `True` and `False` values.
# Fixing the data type will enable us to use mathematical operations on the `is_arrested` column that would not be possible otherwise.

# In[7]:


# Examine the head of the 'is_arrested' column
print(ri.is_arrested.head())

# Check the data type of 'is_arrested'
print(ri.is_arrested.dtype)

# Change the data type of 'is_arrested' to 'bool'
ri['is_arrested'] = ri.is_arrested.astype(bool)

# Check the data type of 'is_arrested' (again)
print(ri.is_arrested.dtype)


# The date and time of each traffic stop are stored in separate columns, both of which are object columns.

# In[8]:


print(ri.iloc[:, 0:4].head())
print(ri.stop_date.dtype, ri.stop_time.dtype)


# We will combine them into a single column and then convert it to a `pandas` `datetime` format. This `datetime` column will function as the `Index` of the dataFrame, that will make it easier to filter and plot it by date.

# In[9]:


# Concatenate 'stop_date' and 'stop_time' (separated by a space)
combined = ri.stop_date.str.cat(ri.stop_time, sep = ' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)

# Examine the data type of 'stop_datetime'
print(ri.stop_datetime.dtype)


# In[10]:


# Set 'stop_datetime' as the index
ri.set_index('stop_datetime', inplace=True)

# Examine the index
print(ri.index)

# Examine the columns ('stop_datetime' is no longer one of the columns)
print(ri.columns)


# Now that we have cleaned the dataset, we can begin analyzing it!

# ### Exploring the relationship between gender and policing

# #### Does the gender of a driver have an impact on police behavior during a traffic stop? In this chapter, we explore that question while practicing filtering, grouping, method chaining, Boolean math, string methods, and more! 

# Before comparing the violations being committed by each gender, we should examine the violations committed by all drivers to get a baseline understanding of the data.

# In[11]:


# Count the unique values in 'violation'
print(ri.violation.value_counts())

print('-------------------------------')

# Express the counts as proportions
print(ri.violation.value_counts(normalize = True))


# Interesting! More than half of all violations are for speeding, followed by other moving violations and equipment violations.

# The question we're trying to answer is whether male and female drivers tend to commit different types of traffic violations.
# In order to answer that, first we create a DataFrame for each gender, and then analyze the violations in each DataFrame separately.

# In[12]:


# Create a DataFrame of female drivers
female = ri[ri.driver_gender == 'F']

# Create a DataFrame of male drivers
male = ri[ri.driver_gender == 'M']

# Compute the violations by female drivers (as proportions)
print(female.violation.value_counts(normalize = True))


print('-------------------------------')

# Compute the violations by male drivers (as proportions)
print(male.violation.value_counts(normalize = True))


# About two-thirds of female traffic stops are for speeding, whereas stops of males are more balanced among the six categories. This doesn't mean that females speed more often than males, however, since we didn't take into account the number of stops or drivers.

# When a driver is pulled over for speeding, many people believe that gender has an impact on whether the driver will receive a ticket or a warning. Can we find evidence of this in the dataset?

# First, we'll create two DataFrames of drivers who were stopped for speeding: one containing females and the other containing males.
# Then, for each gender, we'll use the `stop_outcome` column to calculate what percentage of stops resulted in a *"Citation"* (meaning a ticket) versus a *"Warning"*.

# In[13]:


# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.driver_gender == 'F') & (ri.violation == 'Speeding')]

# Create a DataFrame of male drivers stopped for speeding
male_and_speeding = ri[(ri.driver_gender == 'M') & (ri.violation == 'Speeding')]

# Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize = True))

print('----------------------------------')

# Compute the stop outcomes for male drivers (as proportions)
print(male_and_speeding.stop_outcome.value_counts(normalize = True))


# Interesting! The numbers are similar for males and females: about 95% of stops for speeding result in a ticket. Thus, the data fails to show that gender has an impact on who gets a ticket for speeding.

# During a traffic stop, the police officer sometimes conducts a search of the vehicle. Does the driver's gender affect whether their vehicle is searched? Let's calculate the percentage of all stops that result in a vehicle search, also known as the search rate.

# In[14]:


# Check the data type of 'search_conducted'
print(ri.search_conducted.dtype)

# Calculate the search rate by taking the mean
print(ri.search_conducted.mean())


# It looks like the overall search rate is about 3.8%. Now we compare the rates at which female and male drivers are searched.

# In[15]:


# Calculate the search rate for both groups simultaneously
print(ri.groupby('driver_gender').search_conducted.mean())


# Wow! Male drivers are searched more than twice as often as female drivers. Why might this be?

# Even though the search rate for males is much higher than for females, it's possible that the difference is mostly due to a second factor.
# For example, we might hypothesize that the search rate varies by violation type, and the difference in search rate between males and females is because they tend to commit different violations.
# We can test this hypothesis by examining the search rate for each combination of gender and violation. If the hypothesis was true, we would find that males and females are searched at about the same rate for each violation.

# In[16]:


# Calculate the search rate for each combination of violation and gender
print(ri.groupby(['violation', 'driver_gender']).search_conducted.mean())


# For all types of violations, the search rate is higher for males than for females, disproving our hypothesis.

# During a vehicle search, the police officer may pat down the driver to check if they have a weapon. This is known as a "protective frisk." First, we should check the different types of activities carried out during a search.

# In[17]:


# Count the 'search_type' values
print(ri.search_type.value_counts())


# There were 164 cases where ONLY Protective Frisk was done. In other cases, there were multiple actions taken, resulting in a comma-separated representation of those actions. We can collect all cases when drivers were frisked using a string function.

# In[18]:


# Check if 'search_type' contains the string 'Protective Frisk'
ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na = False)

# Check the data type of 'frisk'
print(ri.frisk.dtype)

# Take the sum of 'frisk'
print(ri.frisk.sum())


# It looks like there were 303 drivers who were frisked. Are males frisked more often than females, perhaps because police officers consider them to be higher risk? Next, we'll examine whether gender affects who is frisked.

# In[19]:


# Create a DataFrame of stops in which a search was conducted
searched = ri[ri.search_conducted == True]

# Calculate the overall frisk rate by taking the mean of 'frisk'
print(searched.frisk.mean())

# Calculate the frisk rate for each gender
print(searched.groupby('driver_gender').frisk.mean())


# Interesting! The frisk rate is higher for males than for females, though we can't conclude that this difference is caused by the driver's gender, as [correlation does not imply causation](https://towardsdatascience.com/correlation-causation-how-alcohol-affects-life-expectancy-a68f7db943f8).

# ### Visual exploratory data analysis

# #### Are you more likely to get arrested at a certain time of day? Are drug-related stops on the rise? In this chapter, we will answer these and other questions by analyzing the dataset visually, since plots can help you to understand trends in a way that examining the raw data cannot. 

# When a police officer stops a driver, a small percentage of those stops ends in an arrest. This is known as the arrest rate. In this part, we'll find out whether the arrest rate varies by time of day.

# In[20]:


# Calculate the overall arrest rate
print(ri.is_arrested.mean())

# Calculate the hourly arrest rate
print(ri.groupby(ri.index.hour).is_arrested.mean())

# Save the hourly arrest rate
hourly_arrest_rate = ri.groupby(ri.index.hour).is_arrested.mean()


# In[21]:


# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Create a line plot of 'hourly_arrest_rate'
hourly_arrest_rate.plot()

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()


# The arrest rate has a significant spike overnight, and then dips in the early morning hours.

# In a small portion of traffic stops, drugs are found in the vehicle during a search. We'll assess whether these drug-related stops are becoming more common over time.
# The Boolean column `drugs_related_stop` indicates whether drugs were found during a given stop. We'll calculate the annual drug rate by resampling this column, and then use a line plot to visualize how the rate has changed over time.
# 

# In[22]:


# Calculate the annual rate of drug-related stops
print(ri.drugs_related_stop.resample('A').mean())

# Save the annual rate of drug-related stops
annual_drug_rate = ri.drugs_related_stop.resample('A').mean()

# Create a line plot of 'annual_drug_rate'
annual_drug_rate.plot()

# Display the plot
plt.show()


# Interesting! The rate of drug-related stops nearly doubled over the course of 10 years. Why might that be the case?

# We might hypothesize that the rate of vehicle searches was also increasing, which would have led to an increase in drug-related stops even if more drivers were not carrying drugs.
# We can test this hypothesis by calculating the annual search rate, and then plotting it against the annual drug rate. If the hypothesis is true, then we'll see both rates increasing over time.

# In[23]:


# Calculate and save the annual search rate
annual_search_rate = ri.search_conducted.resample('A').mean()

# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_rate, annual_search_rate], axis = 'columns')

# Create subplots from 'annual'
annual.plot(subplots = True)

# Display the subplots
plt.show()


# Wow! The rate of drug-related stops increased even though the search rate decreased, disproving our hypothesis.

# The state of Rhode Island is broken into six police districts, also known as zones. How do the zones compare in terms of what violations are caught by police? In this part, we'll create a frequency table to determine how many violations of each type took place in 3 specific zones.

# In[24]:


# Save the frequency table as 'all_zones'
all_zones = pd.crosstab(ri.district, ri.violation)

# Select rows 'Zone K1' through 'Zone K3'
print(all_zones.loc['Zone K1' : 'Zone K3'])

# Save the smaller table as 'k_zones'
k_zones = all_zones.loc['Zone K1' : 'Zone K3']


# In[25]:


# Create a bar plot of 'k_zones'
k_zones.plot(kind = 'bar')

# Display the plot
plt.show()


# In[26]:


# Create a stacked bar plot of 'k_zones'
k_zones.plot(kind = 'bar', stacked = True)

# Display the plot
plt.show()


# Interesting! The vast majority of traffic stops in Zone K1 are for speeding, and Zones K2 and K3 are remarkably similar to one another in terms of violations.

# In the traffic stops dataset, the `stop_duration` column tells us approximately how long the driver was detained by the officer. Unfortunately, the durations are stored as strings, such as `'0-15 Min'`. We have to convert the stop durations to integers. Because the precise durations are not available, we'll have to estimate the numbers using reasonable values:
# *	Convert `'0-15 Min'` to `8`
# *	Convert `'16-30 Min'` to `23`
# *	Convert `'30+ Min'` to `45`

# In[27]:


# Print the unique values in 'stop_duration'
print(ri.stop_duration.unique())

# Create a dictionary that maps strings to integers
mapping = {'0-15 Min' : 8, '16-30 Min' : 23, '30+ Min' : 45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
ri['stop_minutes'] = ri.stop_duration.map(mapping)

# Print the unique values in 'stop_minutes'
print(ri.stop_minutes.unique())


# If you were stopped for a particular violation, how long might you expect to be detained? Let's visualize the average length of time drivers are stopped for each type of violation.

# In[28]:


# Calculate the mean 'stop_minutes' for each value in 'violation_raw'
print(ri.groupby('violation_raw').stop_minutes.mean())

# Save the resulting Series as 'stop_length'
stop_length = ri.groupby('violation_raw').stop_minutes.mean()

# Sort 'stop_length' by its values and create a horizontal bar plot
stop_length.sort_values().plot(kind = 'barh', color = 'blue')

# Display the plot
plt.show()


# ### Analyzing the effect of weather on policing

# #### In this part, we will use a second dataset to explore the impact of weather conditions on police behavior during traffic stops. We perform merging and reshaping datasets, assessing whether a data source is trustworthy, working with categorical data, and other advanced skills. 

# The weather data we'll be using is collected by the National Centers for Environmental Information. In an ideal situation, we could look up the historical weather at the location for each stop. As it is not available, we'll use data from a single weather station near the center of Rhode Island. It is not ideal, but as it is the smallest state, it still could give us a general idea of weather throughout the state.

# In[29]:


# Read 'weather.csv' into a DataFrame named 'weather'
weather = pd.read_csv('data/weather.csv')

weather.head()


# The interpretation of columns is as follows:
# 
# *	`TAVG`, `TMIN`, `TMAX`: Temperature (Fahrenheit)
# *	`AWND`, `WSF2`: Wind speed (miles/hour)
# *	`WT01` ... `WT22`: Bad weather conditions

# First, let's check the temperature columns if we stop any anomaly in the data:

# In[30]:


# Describe the temperature columns
print(weather[['TMIN', 'TAVG', 'TMAX']].describe())

# Create a box plot of the temperature columns
weather[['TMIN', 'TAVG', 'TMAX']].plot(kind = 'box')

# Display the plot
plt.show()


# The `TAVG` values are in between `TMIN` and `TMAX`, and the measurements and ranges seem reasonable.

# We will continue to assess whether the dataset seems trustworthy by plotting the difference between the maximum and minimum temperatures.

# In[31]:


# Create a 'TDIFF' column that represents temperature difference
weather['TDIFF'] = weather.TMAX - weather.TMIN

# Describe the 'TDIFF' column
print(weather.TDIFF.describe())

# Create a histogram with 20 bins to visualize 'TDIFF'
weather.TDIFF.plot(kind = 'hist', bins = 20)

# Display the plot
plt.show()


# The `TDIFF` column has no negative values and its distribution is approximately normal, both of which are signs that the data is trustworthy.

# The `weather` DataFrame contains 20 columns that start with *'WT'*, each of which represents a bad weather condition. For example:
# *	`WT05` indicates "Hail"
# *	`WT11` indicates "High or damaging winds"
# *	`WT17` indicates "Freezing rain"
# For every row in the dataset, each *WT* column contains either a 1 (meaning the condition was present that day) or `NaN` (meaning the condition was not present).
# Let's quantify "how bad" the weather was each day by counting the number of 1 values in each row.

# In[32]:


# Copy 'WT01' through 'WT22' to a new DataFrame
WT = weather.loc[:, 'WT01' : 'WT22']

# Calculate the sum of each row in 'WT'
weather['bad_conditions'] = WT.sum(axis = 'columns')

# Replace missing values in 'bad_conditions' with '0'
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

# Create a histogram to visualize 'bad_conditions'
weather.bad_conditions.plot(kind = 'hist')

# Display the plot
plt.show()


# It looks like many days didn't have any bad weather conditions, and only a small portion of days had more than four bad weather conditions.

# We counted the number of bad weather conditions each day. Now we'll use the counts to create a rating system for the weather.
# The counts range from 0 to 9, and should be converted to ratings as follows:
# *	Convert `0` to `'good'`
# *	Convert `1` through `4` to `'bad'`
# *	Convert `5` through `9` to `'worse'`
# 
# This rating system should make the weather condition data easier to understand.

# In[33]:


# Count the unique values in 'bad_conditions' and sort the index
print(weather.bad_conditions.value_counts().sort_index())

# Create a dictionary that maps integers to strings
mapping = {0:'good', 1:'bad', 2:'bad', 3:'bad', 4:'bad', 5:'worse', 6:'worse', 7:'worse', 8:'worse', 9:'worse'}

# Convert the 'bad_conditions' integers to strings using the 'mapping'
weather['rating'] = weather.bad_conditions.map(mapping)

# Count the unique values in 'rating'
print(weather.rating.value_counts())


# Since the `rating` column only has a few possible values, we'll change its data type to `category` in order to store the data more efficiently. We'll also specify a logical order for the categories, which will be useful for future analyses.

# In[34]:


# Create a list of weather ratings in logical order
cats = ['good', 'bad', 'worse']

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype('category', ordered = True, categories = cats)

# Examine the head of 'rating'
print(weather.rating.head())


# We'll prepare the traffic stop and weather rating DataFrames so that they're ready to be merged.

# In[35]:


# Reset the index of 'ri'
ri.reset_index(inplace = True)

# Examine the head of 'ri'
print(ri.head())

print('------------------------------------------------------------------------------')

# Create a DataFrame from the 'DATE' and 'rating' columns
weather_rating = weather[['DATE', 'rating']]

# Examine the head of 'weather_rating'
print(weather_rating.head())


# The DataFrames will be joined using the `stop_date` column from `ri` and the `DATE` column from `weather_rating`. Thankfully the date formatting matches exactly, which is not always the case!
# Once the merge is complete, we can set s`top_datetime` as the index, which is the column saved in the previous exercise.

# In[36]:


# Examine the shape of 'ri'
print(ri.shape)

# Merge 'ri' and 'weather_rating' using a left join
ri_weather = pd.merge(left=ri, right=weather_rating, left_on='stop_date', right_on='DATE', how='left')

# Examine the shape of 'ri_weather'
print(ri_weather.shape)

# Set 'stop_datetime' as the index of 'ri_weather'
ri_weather.set_index('stop_datetime', inplace=True)


# In the next section, we'll use `ri_weather` to analyze the relationship between weather conditions and police behavior.

# Do police officers arrest drivers more often when the weather is bad? 

# In[37]:


# Calculate the overall arrest rate
print(ri_weather.is_arrested.mean())

print('------------------')

# Calculate the arrest rate for each 'rating'
print(ri_weather.groupby('rating').is_arrested.mean())

print('---------------------------------------')

# Calculate the arrest rate for each 'violation' and 'rating'
print(ri_weather.groupby(['violation', 'rating']).is_arrested.mean())


# Wow! The arrest rate increases as the weather gets worse, and that trend persists across many of the violation types. This doesn't prove a causal link, but it's quite an interesting result!

# Finally, we can look at these statistics by filtering for specific cases with the `.loc[]` accessor.

# In[42]:


# Save the output of the groupby operation
arrest_rate = ri_weather.groupby(['violation', 'rating']).is_arrested.mean()

# Print the 'arrest_rate' Series
print(arrest_rate)

print('----------------------------------------------------------------------------')

# Print the arrest rate for moving violations in bad weather
print('the arrest rate for moving violations in bad weather is ', arrest_rate.loc['Moving violation', 'bad'])

print('----------------------------------------------------------------------------')

# Print the arrest rates for speeding violations in all three weather conditions
print(arrest_rate.loc['Speeding'])


# pandas often gives you more than one way to reach the same result!

# In[39]:


# Unstack the 'arrest_rate' Series into a DataFrame
print(arrest_rate.unstack)

print('-------------------------------------------------')

# Create the same DataFrame using a pivot table
print(ri_weather.pivot_table(index='violation', columns='rating', values='is_arrested'))


# This project is available at https://www.datacamp.com/courses/analyzing-police-activity-with-pandas.
