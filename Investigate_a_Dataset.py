#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate factors that affect whether or not a patient shows up to their medical appointment
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Investigating the data</a></li>
# <li><a href="#eda">Findings and Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > This project will use data collected on 100k medical appointments in Brazil with the aim to analyse what factors influence a patient showing up or not showing up for their medical appointment.
# 
# > Some questions that will be looked at are whether or not patients that are enrolled in the Brasilian welfare program (scholarship) tend to miss their appointments or not. This report will also look at the effect and/or role that age and gender have on showing up for an appointment as well as if receiving an SMS has an effect.

# In[1]:


# import statements for all of the necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Investigating the data

# In[6]:


#load in the data and check out the data
df=pd.read_csv("https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd2e9a_noshowappointments-kagglev2-may-2016/noshowappointments-kagglev2-may-2016.csv")
df.head()


# Following few commands will inspect the data.

# In[ ]:


df.info() #doesn't seem to be any missing values or null values


# In[5]:


df.describe() #age minimum seems strange as it is minus 1


# In[6]:


df.dtypes #data types


# In[7]:


df.nunique() #seems like a number of patients have made appointments more than once


# In[8]:


sum(df.duplicated()) #no duplicated rows


# In[9]:


df.isnull().sum().any() #no null values


# ### Data alterations for better understanding

# I renamed the No-show column because no-show = 'yes' implies an individual did not show up which is a bit confusing. So I changed the column name to skipped appointment and in this case a yes will mean that the indivdual did yes indeed miss the appointment and did not show up.
# The PatientId column could also look better as an integer instead of a float.

# In[3]:


df.rename(columns={"No-show": "Skipped_appointment"}, inplace=True)

df['PatientId'] = df['PatientId'].astype(int)
df.head()


# In[4]:


#change Time data from object to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

#change scheduled day to date without time to be similar to appointment day info
df['ScheduledDay'] = df['ScheduledDay'].dt.date
df['AppointmentDay'] = df['AppointmentDay'].dt.date
df.head(1)


# In[6]:


len(df.query('Age<0')) #only one value less than zero, so not really necessary to remove it from dataset


# In[25]:


df.Handcap.value_counts()
#This variable is difficult to understand. I assume it to mean handicapped or not from the dataset notes but it then has values
#from 0 to 4 attached to it.


# Eliminate certain columns i won't be using in my analysis. I won't be using the Handcap column as its not clear how it is ranked or observed. I will also not use the Alcoholism, Diabetes and Hipertension columns as their averages show that a very low percentage of the dataset have these conditions.

# In[5]:


# Eliminate the columns I won't be using
df.drop(["Hipertension","Diabetes","Alcoholism","Handcap"], axis=1, inplace=True)
df.head(1) #check they have been removed


# <a id='eda'></a>
# ## Analysis and Findings

# The pie chart below shows us that in general majority of the time appointments are not skipped. However, this chart also shows us that roughly a quarter of appointments are skipped, which seems like a pretty high value.

# In[12]:


df.Skipped_appointment.value_counts().plot(kind='pie', title='Percentage of skipped appointments');


# ### Effect of gender on skipping an appointment

# In[14]:


df.Gender.value_counts().plot(kind='pie', title ='Proportion of Females vs. male Patients');


# In[40]:


#creating a function to incorporate the next few lines with regards to variables and their proportions of their totals

def prop_func(df,input1,input2):
    prop_df = len(df.query(input1))/len(df.query(input2))
    print(prop_df)


# In[51]:


#proportion of females that skipped appointments
input1 = '(Gender == "F" and Skipped_appointment == "Yes")'
input2 = '(Gender == "F")'
prop_func(df,input1,input2)


# In[ ]:


#proportion of males that skipped appointments
input1 = '(Gender == "M" and Skipped_appointment == "Yes")'
input2 = '(Gender == "M")'
prop_func(df,input1,input2)


# Although in general there are more females that make appointments than men. Proportional wise there hardly seems to be a huge gender based effect on missing an appointment. For both males and females, the proportion that miss appointments is roughly 20%.

# ### association between number of days between scheduled and appointment day and skipped appointments

# In[53]:


#number of days between the day appointment was scheduled and the actual day of the appointment
df['Diff_in_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df.head(1)


# In[54]:


#skipped appointments vs diff in days between appointment and scheduled days
ax = df.groupby('Skipped_appointment').Diff_in_days.mean().plot(kind='bar',title='Diff in days vs.skipped appointment')
ax.set(ylabel='Difference_in_days');


# Skipped appointments are higher whenever the average number of days between schedule date and appointment day are longer. Average number of days bettween scheduled day and appointment day for skipped appointments is about 16 days and for non skipped appointments it is round about half or 8 days. 
# This could intuitively make sense as people may tend to have something come up last minute or closer to their appointmetn date that they had not foreseen if they booked their appointment way more in advance.

# ### Effect of welfare program 

# In[16]:


schol_skipper= df.groupby("Skipped_appointment").Scholarship.value_counts(normalize=True)
schol_skipper.unstack()


# In[17]:


ax = schol_skipper.unstack().plot(kind='bar', title='Proportion of patients with or without scholarships that skip appointments')
ax.set(ylabel='Proportion of patients');


# This graph seems to show that there isn't really much of a difference caused by receiving welfare between patients skipping or not skipping their appointments.

# In[63]:


ax = df.groupby('Scholarship').Age.mean().plot(kind='bar',title='Scholarship vs.Age')
ax.set(ylabel='Average age');


# The average age of people that receive welfare is younger than those that do not.

# ### Role of Age on appointments

# In[53]:


#proportion of missed appointments by children
input1 = '(Age <= 10 and Skipped_appointment == "Yes")'
input2 = '(Skipped_appointment == "Yes")'
prop_func(df,input1,input2)


# In[3]:


df.Gender.value_counts().plot(kind='pie', title ='Proportion of Females vs. male Patients');


# In[14]:


#histogram of the ages
df['Age'].plot.hist(alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Graph of Patients ages');


# Interesting enough, this graph shows that alot of the appointments are for those that are younger than 10 years old, as this is where the bar has its highest frequency. Normally children do not attend appointments alone and the proportion of missed appointments by children under 10 is about 16%. Thus this gives light to many questions behind why this is so, as this may be due to the child or moreso the parent or guardian or whoever is supposed to take them.

# In[18]:


#average ages of those that skip or did not skip appointments
ax = df.groupby('Skipped_appointment').Age.mean().plot(kind='bar', title='Average Age vs. skipped appointments');
ax.set(ylabel='Average age');


# This graph shows that the average ages of people that do not miss appointments is slightly higher than the average age of people that do miss appointments.

# In[7]:


#creating bins for age groups to see which age groups skip appointments more than others
df['Age_group'] = pd.cut(df.Age,[0,18,35,50,65,100], labels=['0-18','18-35','35-50', '50-65','65+'])
df['Age_group'].head()


# In[8]:


Age_group_skips = df.groupby('Skipped_appointment').Age_group.value_counts(normalize=True)
ax = Age_group_skips.unstack().plot(kind='bar', title='Skipped appointments according to Age group');
ax.set(ylabel='Number of patients');
#schol_skipper= df.groupby("Skipped_appointment").Scholarship.value_counts(normalize=True)


# This graph shows that the youngest age group of 0-18, skips appointments the least followed by the 50-65 age group. However with regards to skipped appointments the young category also has the 2nd highest skipped appointments with the oldest category above 65 skipping the least amount of appointments.

# ### Effect of an SMS received on attending an appointment

# In[ ]:


df.SMS_received.mean() #average sms's sent out


# In[54]:


#proportion of people that received an SMS and also made it to their appointment
input1 = '(SMS_received == "1" and Skipped_appointment == "No")'
input2 = '(SMS_received == "1")'
prop_func(df,input1,input2)


# Only about 32% of patients receive an SMS about an appointment and this seems like too low a value, perhaps more patients would not miss an appointment if they all managed to receive an SMS or have access to being reminded.
# 
# This can also be seen in the fact that out of all the patients that did receive an SMS, 72% of them did not miss their appointments.

# <a id='conclusions'></a>
# ## Conclusions
# 
# > To conclude, it can be noted that quite a few reasons can be behind why a patient misses a medical appointment. With the limited data given it seems like receiving an SMS has a role to play with regards to missing an appointment. However the other factors considered in this report do not seem to really give a definitive answer as to their effects on making an appointment as for many of the comparisons make there does not seem to be much of a difference between the values arrived at for missing an appointment or not missing one.
# 
# > Further, more in depth analysis and data would need to be provided and then undertaken to be better able to understand what may be the reasons behind missed appointments. For example data on distance to nearest hospital may be helpful as well as information on what the reason behind the medical visit is.

# In[15]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

