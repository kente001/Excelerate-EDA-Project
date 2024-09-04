#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[3]:


df = pd.read_csv('C:/Users/User/Desktop/Execelerate data set/Revised Opportunity Wise.csv')
df.head(5)


# In[5]:


df.dtypes


# In[9]:


# Count of unique ids
sponsor_combination_count = df['Opportunity Id'].nunique()
sponsor_combination_count


# In[58]:


# Count of unique ids
sponsor_combination_count = df['Profile Id'].nunique()
sponsor_combination_count


# In[62]:


# Count occurrences of each unique combination of sponsors
sponsor_combination_count = df['Profile Id'].value_counts()
sponsor_combination_count.head(5)


# In[23]:


sponsor_combination_count = df['Status Description'].value_counts(normalize=True)*100
round(sponsor_combination_count.head(5),2)


# In[26]:


Reward_Amount_description = df['Skill Points Earned'].describe()
Reward_Amount_description


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt



# Count the occurrences of each gender category
gender_counts = df['Gender'].value_counts()

# Plot the bar graph
ax=gender_counts.plot(kind='bar', color=['blue', 'red', 'green', 'purple'])

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.xticks(rotation=45)  # Rotate the x labels for better readability

# Add values at the top of each bar
for container in ax.containers:
    ax.bar_label(container, label_type='edge')

# Show the plot
plt.show()


# In[32]:


df.head(5)


# In[35]:





# Count the frequency of each year
_counts = df['Current Student Status'].value_counts()

# Plot the bar chart for the frequency of each year
ax = _counts.plot(kind='bar', color='blue')

# Add labels and title
plt.xlabel('Current Student Status')
plt.ylabel('count')
plt.title('Frequency of Current Student Status')

# Add values at the top of each bar
for container in ax.containers:
    ax.bar_label(container, label_type='edge')

# Show the plot
plt.show()


# In[49]:


# Count occurrences of each gender
gender_counts = df['Gender'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff', '#99ff99'])
plt.title('Percentage Gender Distribution')
plt.show()


# In[41]:


opportunity_counts = df['Opportunity Category'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(opportunity_counts, labels= opportunity_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff', '#99ff99'])
plt.title('Percentage Opportunity Distribution')
plt.show()


# In[44]:


# Plot the histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df['Reward Amount'], bins=10, edgecolor='black', color='blue')

# Add values on top of the bars
for count, bin_edge in zip(n, bins):
    # Get the x-coordinate for the middle of each bar
    x = (bin_edge + bins[bins.searchsorted(bin_edge + 1)]) / 2
    plt.text(x, count, int(count), ha='center', va='bottom', fontsize=10, color='black')

plt.title('Histogram of Reward Amount')
plt.xlabel('Reward Amount')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[45]:


# Plot the histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df['Skill Points Earned'], bins=10, edgecolor='black', color='blue')

# Add values on top of the bars
for count, bin_edge in zip(n, bins):
    # Get the x-coordinate for the middle of each bar
    x = (bin_edge + bins[bins.searchsorted(bin_edge + 1)]) / 2
    plt.text(x, count, int(count), ha='center', va='bottom', fontsize=10, color='black')

plt.title('Histogram of Skill Points Earned')
plt.xlabel('Skill Points Earned')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[48]:



# Count occurrences of each badge name
badge_counts = df['Badge Name'].value_counts()

# Get the top five badge names
top_five_badges = badge_counts.head(5)

# Plot the bar chart
plt.figure(figsize=(10, 6))
ax = top_five_badges.plot(kind='bar', color='blue', edgecolor='black')

# Add values on top of the bars
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, color='black')

plt.title('Distribution of Top Five Badge Names')
plt.xlabel('Badge Name')
plt.ylabel('Frequency of top 5 Badge Names')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[56]:


sponsor_combination_count = df['Apply Date'].count()
round(sponsor_combination_count,2)


# In[60]:


sponsor_combination_count = df['Apply Date'].value_counts(normalize=True)*100
round(sponsor_combination_count,2)


# In[61]:


duplicates = df.duplicated()

print(duplicates)


# In[69]:


filtered_df = df[df['Badge Id'].notnull()][['Profile Id','Badge Id']]

print(filtered_df)


# In[4]:




# Create a new column to differentiate between status and badge
df['Badge Status'] = df.apply(lambda row: 'Awarded' if row['Status Description'] == 'REWARDS AWARD' else row['Status Description'], axis=1)

# Optionally, fill the 'badge_name' column with a descriptive value for non-award rows
df['Badge Name'].fillna('No Badge', inplace=True)

df.head(5)


# In[79]:


not_started_or_rejected = df[df['Status Description'].isin(['Not Started', 'Rejected'])]
factors = not_started_or_rejected.groupby(['Status Description', 'Gender', 'Current Student Status']).size().unstack(fill_value=0)
factors


# In[82]:


not_started_or_rejected = df[df['Status Description'].isin(['Not Started', 'Rejected'])]
not_started_or_rejected['Current Student Status'].value_counts()


# In[84]:


rejected = df[df['Status Description'] == 'Rejected']
rejected['Current Student Status'].value_counts()


# In[88]:


team_allocated = df[df['Status Description'] == 'Team Allocated']
team_allocated['Current Student Status'].value_counts()


# In[103]:


not_started_or_rejected = df[df['Status Description'] == 'Rewards Award']
not_started_or_rejected[['Skill Points Earned', 'Reward Amount']].head(20)


# In[110]:


# Filter rows where the applicant gained skill points and received a badge, but no reward amount
filtered_df = df[
    (df['Skill Points Earned'] > 0) &                    # Check if skill points are greater than 0
    (df['Badge Name'] != 'No Badge') &                       # Check if Badge Name is not null (assuming a badge indicates success)
    (df['Reward Amount'].isnull())                       # Check if Reward Amount is null
]

# Display the result
filtered_df


# In[113]:


filtered_df = df[
    (df['Skill Points Earned'] > 0) &                    
                                                         
    (df['Badge Name'] == 'No Badge')                      
]

# Display the result
filtered_df


# In[24]:


Have_badge = df[df['Badge Status'] == 'Team Allocated']
user_program_counts = Have_badge.groupby('Profile Id')['Opportunity Id'].nunique()

# Step 3: Filter users who have applied to multiple programs
users_multiple_programs = user_program_counts[user_program_counts > 1].index

# Step 4: Get details of these users
details_multiple_programs = Have_badge[Have_badge['Profile Id'].isin(users_multiple_programs)]

# Display the results
final = details_multiple_programs.sort_values(by='Profile Id', ascending = True)[['Profile Id','Opportunity Id','Status Description']]
final.head(20)


# In[121]:


user_program_counts =df.groupby('Profile Id')['Opportunity Id'].nunique()

# Step 3: Filter users who have applied to multiple programs
users_multiple_programs = user_program_counts[user_program_counts > 1]..index

# Step 4: Get details of these users
details_multiple_programs = df[df['Profile Id'].isin(users_multiple_programs)]

# Display the results
final = details_multiple_programs.sort_values(by='Profile Id', ascending = True)
[['Profile Id','Opportunity Id','Gender','Current Student Status','Status Description']]
final.head(100)


# In[116]:


Team_Allocated = final[final['Status Description'] == 'Team Allocated']
Team_Allocated['Current Student Status'].value_counts()


# In[35]:


Rejected = final[final['Status Description'] == 'Rejected']
Rejected['Current Student Status'].value_counts()


# In[36]:


Rewarded = final[final['Status Description'] == 'Rewards Award']
Rewarded['Current Student Status'].value_counts()


# In[39]:


pd.set_option('display.max_columns', None)
summary_statistics = final.describe(include='all')
summary_statistics


# In[122]:


counts =df.groupby('Profile Id')['Opportunity Id'].nunique()
counts
# Step 3: Filter users who have applied to multiple programs
users_single_program = counts[user_program_counts == 1].index

# Step 4: Get details of these users
details_single_program = df[df['Profile Id'].isin(users_single_program)]

# Display the results
#finals = details_single_program.sort_values(by='Profile Id', ascending = True)
[['Profile Id','Opportunity Id','Gender','Current Student Status','Status Description']]
finals.head(10)


# In[41]:


pd.set_option('display.max_columns', None)
summary_statistics = finals.describe(include='all')
summary_statistics


# In[43]:


filt = finals['Badge Status'].value_counts()
filt


# In[44]:


filt = final['Badge Status'].value_counts()
filt


# In[57]:


total = final['Current Student Status'].value_counts()
total


# In[115]:


countries_df = df.drop_duplicates(subset='Profile Id')
countries_df = countries_df['Country'].value_counts()
print('Top 10 Countries:') 
countries_df.head(10)


# In[79]:


unique_profile_df = final.drop_duplicates(subset='Profile Id')
unique_profile = unique_profile_df['Gender'].value_counts()
unique_profile


# In[69]:


profile = final['Profile Id'].unique()
filt = final[final['Profile Id'].isin(profile)]
total = filt['Gender'].value_counts()
filt


# In[46]:


filt = finals['Gender'].value_counts()
filt


# In[67]:


num_many = final['Profile Id'].nunique()
num_many


# In[63]:


num_single = finals['Profile Id'].count()
num_single


# In[82]:


num_single = finals['Current Student Status'].value_counts(normalize = True)*100
round(num_single,2)


# In[81]:


num_many = unique_profile_df['Current Student Status'].value_counts(normalize = True)*100
round(num_many,2)


# In[83]:


num_single = finals['Gender'].value_counts(normalize = True)*100
round(num_single,2)


# In[84]:


num_many = unique_profile_df['Gender'].value_counts(normalize = True)*100
round(num_many,2)


# In[86]:




# Aggregate by profile id
df_aggregated = df.groupby('Profile Id').agg({
    'Skill Points Earned': 'sum',  # Sum skill points for each profile
    'Reward Amount': 'sum',        # Sum reward amounts for each profile
    'Badge Status': lambda x: 'Earned' if 'Earned' in x.values else 'None'  # Determine if badge was earned
}).reset_index()

# Create binary flag for badge earned
df_aggregated['Badge Earned'] = df_aggregated['Badge Status'].apply(lambda x: 1 if x == 'Earned' else 0)
df_aggregated


# In[87]:






# In[88]:


impact_summary = df_aggregated.groupby('Badge Earned').agg({
    'Skill Points Earned': 'mean',
    'Reward Amount': 'mean'
}).reset_index()

impact_summary


# In[89]:


from scipy.stats import ttest_ind

# Split data into two groups
badge_earned = df_aggregated[df_aggregated['Badge Earned'] == 1]
no_badge = df_aggregated[df_aggregated['Badge Earned'] == 0]

# T-test for Skill Points
t_stat, p_value = ttest_ind(badge_earned['Skill Points Earned'], no_badge['Skill Points Earned'])
print(f'T-test for Skill Points: t-statistic = {t_stat}, p-value = {p_value}')

# T-test for Reward Amount
t_stat, p_value = ttest_ind(badge_earned['Reward Amount'], no_badge['Reward Amount'])
print(f'T-test for Reward Amount: t-statistic = {t_stat}, p-value = {p_value}')


# In[5]:


df['Points Missing'] = df['Skill Points Earned'].isnull()
df['Reward Missing'] = df['Reward Amount'].isnull()
df.head(5)


# In[93]:


df_aggregated = df.groupby('Profile Id').agg({
    'Skill Points Earned': 'sum',       # Sum up skill points
    'Reward Amount': 'sum',      # Sum up reward amounts
    'Badge Status': 'max'        # Max value to check if at least one badge was awarded
}).reset_index()

df_aggregated['BadgeAwarded'] = df_aggregated['Badge Status'].apply(lambda x: 1 if x == 'Rewards Award' else 0)
df_aggregated


# In[95]:


df_success = df_aggregated[df_aggregated['BadgeAwarded'] == 1]
df_success


# In[96]:


df_no_success = df_aggregated[df_aggregated['BadgeAwarded'] == 0]
df_no_success


# In[99]:


avg_skill_points_success = df_success['Skill Points Earned'].mean()
avg_skill_points_no_success = df_no_success['Skill Points Earned'].mean()

avg_reward_success = df_success['Reward Amount'].mean()
avg_reward_no_success = df_no_success['Reward Amount'].mean()


# In[100]:


avg_skill_points_success


# In[101]:


avg_skill_points_no_success


# In[102]:


avg_reward_success


# In[103]:


avg_reward_no_success


# In[105]:


from scipy.stats import ttest_ind

skill_points_success = df_success['Skill Points Earned']
skill_points_no_success = df_no_success['Skill Points Earned']

reward_success = df_success['Reward Amount']
reward_no_success = df_no_success['Reward Amount']

t_stat_skill, p_value_skill = ttest_ind(skill_points_success, skill_points_no_success, nan_policy='omit')
t_stat_reward, p_value_reward = ttest_ind(reward_success, reward_no_success, nan_policy='omit')

print(f"Skill Points Earned T-statistic: {t_stat_skill}, P-value: {p_value_skill}")
print(f"Reward Amount T-statistic: {t_stat_reward}, P-value: {p_value_reward}")


# In[6]:


df.to_csv('C:/Users/User/Downloads/Final Opportunity Wise.csv', index=False)


# In[1]:


import pandas as pd 


# In[2]:


df = pd.read_csv('C:/Users/User/Downloads/Final Opportunity Wise.csv')


# In[3]:


df


# In[52]:


#Total number of applicants 
df['Profile Id'].nunique()


# In[53]:


# total number of applicants accepted
filt = df[df['Badge Status'] == 'Rejected']
filt['Profile Id'].nunique()


# In[54]:


# total number of learners whose application is yet to be consindered 
applied_filt  = df[df['Badge Status'] == 'Applied']
applied_filt['Profile Id'].nunique()


# In[55]:


# total number of learners who withdrew from the program 
withdraw_filt  = df[df['Badge Status'] == 'Withdraw']
withdraw_filt['Profile Id'].nunique()


# In[40]:


# countries with highest number of withdraws
unique_withdrawn_df = withdraw_filt.drop_duplicates(subset=['Profile Id'])
unique_withdrawn_df['Country'].value_counts()


# In[56]:


Top_countries = df.drop_duplicates(subset=['Profile Id'])
top =Top_countries['Country'].value_counts()
top.head(10)


# In[58]:


replacements = {
    'Saintlouis': 'Saint Louis',
    'Stlouis': 'Saint Louis'}
df['City'] = df['City'].replace(replacements)

Us_applicants = df.drop_duplicates(subset=['Profile Id'])
Us_applicants = Us_applicants[Us_applicants['Country'] == 'United States']
Top_city = Us_applicants['City'].value_counts()
Top_city.head(10)


# In[34]:


# how many applicants fron US withdrew from the program
us_withdraw = Us_applicants[Us_applicants['Badge Status'] == 'Withdraw']
us_withdraw['Profile Id'].nunique()


# In[60]:


opportunity_popularity =df['Opportunity Category'].value_counts()
opportunity_popularity.head(10)


# In[62]:


completed = df[df['Badge Status'] == 'Rewards Award']
popular_completed = completed['Opportunity Category'].value_counts()
popular_completed


# In[64]:


completed = df[df['Badge Status'] == 'Rewards Award']
completed['Opportunity Category'].count()


# In[65]:


# accepted opportunity
accepted_opportunity = df[df['Badge Status'] != 'Rejected']
accepted_opportunity['Opportunity Category'].count()


# In[66]:


rejected_opportunity = df[df['Badge Status'] == 'Rejected']
rejected_opportunity['Opportunity Category'].count()


# In[ ]:


rejected_opportunity = df[(df['Badge Status'] == 'Team Allocated') & (df['Badge Status'] == 'Not Started')


# In[68]:


df['Badge Status'].unique


# In[4]:


import pandas as pd
import numpy as np



# Step 1: Exclude rows where both columns are null (indicating learners who haven't started)
filtered_df = df.dropna(subset=['Reward Amount', 'Skill Points Earned'], how='all')

# Step 2: Apply IQR method to detect outliers

# Function to detect outliers using IQR
def detect_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)

# Check for outliers in 'reward_amount'
outliers_reward = detect_outliers_iqr(filtered_df['Reward Amount'])

# Check for outliers in 'skills_points_earned'
outliers_skills = detect_outliers_iqr(filtered_df['Skill Points Earned'])

# Display rows identified as outliers
outliers_df = filtered_df[outliers_reward | outliers_skills]
outliers_df


# In[74]:


import matplotlib.pyplot as plt

# Boxplot to visualize 'reward_amount'
filtered_df['Reward Amount'].plot(kind='box')
plt.show()

# Boxplot to visualize 'skills_points_earned'
filtered_df['Skill Points Earned'].plot(kind='box')
plt.show()


# In[69]:


df.head(5)


# In[75]:


import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'reward_amount': [100, 200, 150, np.nan, 300, 500, np.nan, 1000, np.nan],
    'skills_points_earned': [10, 20, np.nan, 15, 50, 5, np.nan, 100, np.nan]
}

df2 = pd.DataFrame(data)

# Step 1: Exclude rows where both columns are null
filtered_df2 = df2.dropna(subset=['reward_amount', 'skills_points_earned'], how='all')

# Step 2: Apply IQR method to detect outliers
def detect_outliers_iqr2(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    return (column < lower_bound) | (column > upper_bound)

# Check for outliers in 'reward_amount'
outliers_reward2 = detect_outliers_iqr2(filtered_df2['reward_amount'])

# Check for outliers in 'skills_points_earned'
outliers_skills2 = detect_outliers_iqr2(filtered_df2['skills_points_earned'])

# Display rows identified as outliers
outliers_df2 = filtered_df2[outliers_reward2 | outliers_skills2]
outliers_df2


# In[8]:


Q1 = df['Reward Amount'].quantile(0.25)
Q3 = df['Reward Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
lower_bound


# In[9]:


upper_bound


# In[10]:


Q1


# In[11]:


IQR


# In[14]:


# checking for outliers in the Reward amount column
import pandas as pd


filt_ = df.dropna(subset=['Reward Amount', 'Skill Points Earned'], how='all')

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for 'reward_amount'
Q1 = filt_['Reward Amount'].quantile(0.25)
Q3 = filt_['Reward Amount'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Determine the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

# Identify outliers
outliers = filt_[(filt_['Reward Amount'] < lower_bound) | (filt_['Reward Amount'] > upper_bound)]
print("\nOutliers in 'Reward Amount':")
outliers


# In[15]:


filt_ = df.dropna(subset=['Reward Amount', 'Skill Points Earned'], how='all')

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for 'reward_amount'
Q1 = filt_['Skill Points Earned'].quantile(0.25)
Q3 = filt_['Skill Points Earned'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Determine the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

# Identify outliers
s_outliers = filt_[(filt_['Skill Points Earned'] < lower_bound) | (filt_['Skill Points Earned'] > upper_bound)]
print("\nOutliers in 'Skill Points Earned':")
s_outliers


# In[21]:


df[df['Skill Points Earned'] > 0]


# In[4]:


median_value = df['Skill Points Earned'].median()
# Replace values in the range 10 to 1776 with the median
df['Skill Points Earned'] = df['Skill Points Earned'].apply(lambda x: median_value if 10 <= x <= 1776 else x)
df


# In[22]:


#T-Test for Gender and Badge Earning Rate
from scipy.stats import ttest_ind

# Filter the data for badge earning (0 for not earned, 1 for earned)
df['Badge Earned'] = df['Badge Status'].apply(lambda x: 1 if x == 'Rewards Award' else 0)

# Split data by gender
male_badge_earned = df[df['Gender'] == 'Male']['Badge Earned']
female_badge_earned = df[df['Gender'] == 'Female']['Badge Earned']

# Perform independent t-test
t_stat, p_value = ttest_ind(male_badge_earned, female_badge_earned, equal_var=False)

print(f"T-Statistic: {t_stat}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("The difference in badge earning rates between genders is statistically significant.")
else:
    print("There is no statistically significant difference in badge earning rates between genders.")


# In[27]:


gender_df = df.drop_duplicates(subset=['Profile Id'])
gender_df1 = gender_df[(gender_df['Gender']== 'Male')| (gender_df['Gender']== 'Female')]
gender_df2 = gender_df1['Gender'].value_counts()
gender_df2


# In[28]:


# After running the first t-test it did show that males have a higher rate of completing the programs, after checking the number I
# suspect that this was due to higher number of male applicants. This resulted to downsampling to take care of higher samples 
# for the males. 
#The downsampling approach has helped confirm that the difference is real and not just an artifact of group size imbalance.
from sklearn.utils import resample
from scipy.stats import ttest_ind

# Separate the male and female data
males = df[df['Gender'] == 'Male']
females = df[df['Gender'] == 'Female']

# Downsample males to match the number of females
males_downsampled = resample(males, replace=False, n_samples=len(females), random_state=42)

# Combine the balanced dataset
balanced_df = pd.concat([males_downsampled, females])

# Conduct the t-test on the balanced data
male_badge_earned = balanced_df[balanced_df['Gender'] == 'Male']['Badge Earned']
female_badge_earned = balanced_df[balanced_df['Gender'] == 'Female']['Badge Earned']

# Perform independent t-test
t_stat, p_value = ttest_ind(male_badge_earned, female_badge_earned, equal_var=False)

print(f"T-Statistic: {t_stat}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("The difference in badge earning rates between genders is statistically significant.")
else:
    print("There is no statistically significant difference in badge earning rates between genders.")


# In[130]:


# success rate by program
# Define the conditions to filter out rejected, non-starters, withdrawals, and those still in progress
conditions_to_exclude = [
    'Team Allocated',  # In progress
    'Rejected',        # Rejected applicants
    'Applied',
    'Started' # application not confirmed the program
]

# Filter out the excluded conditions
filtered_df = df[~df['Badge Status'].isin(conditions_to_exclude)]
filtered_df1 = filtered_df[filtered_df['Badge Name'] != 'No Badge']
success_rate = filtered_df1.groupby('Opportunity Name')['Badge Status'].size()/filtered_df.groupby('Opportunity Name')['Badge Status'].size() 
# Step 3: Convert to a percentage
success_rate = success_rate.fillna(0) * 100

# Display the success rates
success_rate = success_rate.reset_index(name='SuccessRate (%)')
success_rate = success_rate.sort_values(by='SuccessRate (%)', ascending=False)
print('Top 10 Programs with highest Success by percentage:')
round(success_rate.head(10),2)




# In[91]:


# overall success rate
# Define the conditions to filter out rejected, non-starters, withdrawals, and those still in progress
conditions_to_exclude = conditions_to_exclude = [
    'Team Allocated',  # In progress
    'Rejected',        # Rejected applicants
    'Applied',
    'Started' # application not confirmed the program
]

    
    # application not confirmed the program


# Filter out the excluded conditions
success_df = df[~df['Badge Status'].isin(conditions_to_exclude)]
success_df1 = success_df[filtered_df['Badge Name'] != 'No Badge']
success_rate = success_df1['Opportunity Name'].count()/success_df['Opportunity Name'].count() 
# Step 3: Convert to a percentage
success_rate  = success_rate*100
success_rate


# In[87]:


#Failure rate
# Define the conditions to filter out rejected, non-starters, withdrawals, and those still in progress
conditions_to_exclude = [
    'Team Allocated',  # In progress
    'Rejected',        # Rejected applicants
    'Applied',
    'Started',
    # application not confirmed the program
]

# Filter out the excluded conditions
filtered_df = df[~df['Badge Status'].isin(conditions_to_exclude)]
filtered_df1 = filtered_df[filtered_df['Badge Name'] == 'No Badge']
failure_rate = filtered_df1['Opportunity Name'].count()/filtered_df['Opportunity Name'].count() 
# Step 3: Convert to a percentage
failure_rate  = failure_rate*100
failure_rate




# In[118]:


#Acceptance Rate by program
# Define the conditions to filter out rejected, non-starters, withdrawals, and those still in progress
conditions_to_exclude = [
      'Rejected',        # Rejected applicants
         'Applied'# application not confirmed the program
]

# Filter out the excluded conditions
accepted_df = df[~df['Badge Status'].isin(conditions_to_exclude)]
Applied_df = df[df['Badge Status'] != 'Applied']
Acceptance_rate = accepted_df.groupby('Opportunity Name')['Profile Id'].size()/Applied_df.groupby('Opportunity Name')['Profile Id'].size() 
# Step 3: Convert to a percentage
Acceptance_rate = Acceptance_rate.fillna(0) * 100

# Display the success rates
Acceptance_rate = Acceptance_rate.reset_index(name='AcceptanceRate (%)')
Acceptance_rate = Acceptance_rate.sort_values(by='AcceptanceRate (%)', ascending=False)
print('Acceptance rate for all programs by percentage:')
Acceptance_rate.tail(5)




# In[117]:


#Overall Acceptance Rate 
# Define the conditions to filter out rejected, non-starters, withdrawals, and those still in progress
conditions_to_exclude = [
      'Rejected',        # Rejected applicants
         'Applied'# application not confirmed the program
]

Applied_df = df[df['Badge Status'] != 'Applied']

# Filter out the excluded conditions
accepted_df = df[~df['Badge Status'].isin(conditions_to_exclude)]
Acceptance_rate = accepted_df['Opportunity Name'].count()/Applied_df['Opportunity Name'].count() 
# Step 3: Convert to a percentage
Acceptance_rate = Acceptance_rate * 100

print('Acceptance rate for all programs by percentage:')
round(Acceptance_rate,2)


# In[129]:


fil1 =accepted_df.groupby('Opportunity Name')['Profile Id'].size()
fil1 = fil1.reset_index(name='Count_Opportunity')
fil1 = fil1.sort_values(by= 'Count_Opportunity', ascending = False)
print('Top 5 Programs with highest Applicants:')
fil1.head(5)


# In[99]:


#Rejection Rate by program
# Define the conditions to filter out rejected, non-starters, withdrawals, and those still in progress
conditions_to_exclude = [
      'Rejected',        # Rejected applicants
         # application not confirmed the program
]

# Filter out the excluded conditions
rejection_df = df[df['Badge Status'].isin(conditions_to_exclude)]
Applied_df = df[df['Badge Status'] != 'Applied']
rejection_rate = rejection_df.groupby('Opportunity Name')['Profile Id'].size()/Applied_df.groupby('Opportunity Name')['Profile Id'].size() 
# Step 3: Convert to a percentage
rejection_rate = rejection_rate.fillna(0) * 100

# Display the success rates
rejection_rate = rejection_rate.reset_index(name='RejectionRate (%)')
rejection_rate = rejection_rate.sort_values(by='RejectionRate (%)', ascending=False)
rejection_rate


# In[119]:


#Rejection Rate by program
# Define the conditions to filter out rejected, non-starters, withdrawals, and those still in progress
conditions_to_exclude = [
      'Rejected',        # Rejected applicants
         # application not confirmed the program
]

# Filter out the excluded conditions
rejection_df = df[df['Badge Status'].isin(conditions_to_exclude)]
Applied_df = df[df['Badge Status'] != 'Applied']
rejection_rate = rejection_df['Opportunity Name'].count()/Applied_df['Opportunity Name'].count() 
# Step 3: Convert to a percentage
rejection_rate = rejection_rate * 100
print('Rejection Rate for all Programs: ')
round(rejection_rate,2)


# In[105]:


number = df['Profile Id'].count()
print('Total number of applications:',  number)


# In[141]:


df['Apply Date'] = pd.to_datetime(df['Apply Date'])
monthly_trend = df.resample('M', on='Apply Date').size()
print('Applications per Month')
monthly_trend


# In[139]:


# Group by year and month
monthly_trend = df.groupby(df['Apply Date'].dt.to_period('Y')).size()
print('Applications per Year')
monthly_trend


# In[142]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
monthly_trend.plot(kind='line', marker='o')
plt.title('Course Applications Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Applications')
plt.grid(True)
plt.show()


# In[148]:


df = df[(df['Gender'] == 'Male') | (df['Gender'] == 'Female') ]
trend_by_course = df.groupby(['Gender', df['Apply Date'].dt.to_period('M')]).size().unstack().transpose()
trend_by_course.plot(kind='line', figsize=(12, 8))
plt.title('Course Applications Trend by Gender')
plt.xlabel('Month')
plt.ylabel('Number of Applications')
plt.show()


# In[143]:


df.head(5)


# In[149]:


import matplotlib.pyplot as plt

monthly_trend.plot(kind='line', marker='o')
plt.title('Monthly Trend of Course Applications')
plt.xlabel('Month')
plt.ylabel('Number of Applications')
plt.show()


# In[14]:


df.to_csv('C:/Users/User/Downloads/Final_Opportunity Wise.csv', index=False)


# In[13]:


df.dtypes


# In[7]:


df['Apply Date'] = pd.to_datetime(df['Apply Date'])


# In[12]:


df['Opportunity End Date '] = pd.to_datetime(df['Opportunity End Date'])


# In[18]:


filt= df[df['Badge Status'] == 'Rewards Award']
success = filt.groupby('Opportunity Name')['Badge Status'].count()
success


# In[12]:


df_dropout = df[(df['Badge Status'] == 'Dropped Out') | (df['Badge Status'] == 'Withdraw')]

df_dropout['Gender'].value_counts(normalize = True)*100


# In[7]:


df['Reward Amount'].sum()


# In[21]:


import pandas as pd
df3 = pd.read_csv('C:/Users/User/Downloads/src_offline_buildings_202408231641.csv')


# In[ ]:





# In[22]:


import pandas as pd




# Convert the Date column to datetime if it's not already
df3['run_date'] = pd.to_datetime(df3['run_date'])

# Pivot the DataFrame
pivot_df = df3.pivot_table(index='building', columns='run_date', values='onu_count', aggfunc='sum').fillna(0)

# Optional: Rename columns to show dates as strings
pivot_df.columns = pivot_df.columns.strftime('%Y-%m-%d')

# Reset index if you want 'Building Name' as a column instead of index
pivot_df.reset_index(inplace=True)

pivot_df


# In[20]:


total_offline_per_building = df3.groupby('building')['onu_count'].sum().reset_index()
total_offline_per_building.columns = ['Building Name', 'Total Routers Off']
total_offline_per_building


# In[25]:


days_with_offline_routers = df3[df3['onu_count'] > 0].groupby('building')['run_date'].nunique().reset_index()
days_with_offline_routers.columns = ['Building Name', 'Days with Offline Routers']

total_days = df3['run_date'].nunique()

days_with_offline_routers['Offline Probability'] = days_with_offline_routers['Days with Offline Routers'] / total_days
days_with_offline_routers


# In[26]:


analysis_df = pd.merge(total_offline_per_building, days_with_offline_routers, on='Building Name', how='left')
analysis_df.fillna(0, inplace=True)  # In case some buildings have no offline days
analysis_df.sort_values(by=['Offline Probability', 'Total Routers Off'], ascending=False, inplace=True)

analysis_df


# In[29]:


df3.groupby('building')['onu_count'].sum()


# In[ ]:


analysis_df2 = analysis_df[analysis_df['Offline Probability']]

