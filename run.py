import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Loading data
data = pd.read_csv('data/data.csv',encoding= 'unicode_escape')
feature_desc = pd.read_csv('data/feature_descriptions.csv', encoding= 'unicode_escape')


#Part 1

# Removing columns with more than 45% missing values
data.dropna(thresh=0.55*len(data), axis=1, inplace=True)
target = data['TARGET']
data.drop(columns='TARGET',inplace=True)
# print(data)

# missing_vals = data.isna().sum()
# missing_vals_percent = missing_vals / len(data) * 100
# missing_df = pd.DataFrame({'missing_vals': missing_vals, 'missing_percent': missing_vals_percent})
# print(missing_df)

# Imputing missing values using the median
median_vals = data.median(numeric_only=True)
data.fillna(median_vals, inplace=True)

# I am using the median to impute the missing data because it is a 
# good approximation of the data and it is less affected by skewed data
# compared to the mean.


# Part 2


# Check skewness of numerical features
skewness = data.skew(numeric_only=True)

# Plot histograms of skew features
skew_feats = skewness[abs(skewness) > 0.5]
skew_feats_list = skew_feats.index.tolist()
data.hist(column=skew_feats_list, figsize=(15, 15))
plt.savefig('plots/Histograms of Skewed Feats.png')
plt.clf()


transformed_data = data
transformed_feats = []
for feature in skew_feats_list:
    plt.hist(data[feature], bins=50)
    plt.title('Original Histogram of {}'.format(feature))
    plt.savefig('plots/Original Histogram of {}.png'.format(feature))
    plt.clf()


    try:
        # try a log transformation
        transformed = np.log(data[feature])
        skewness = stats.skew(transformed.dropna())
        # print(f"Log transform: {feature}, skewness: {skewness}")
    except:
        pass
    
    if not abs(skewness) < 0.5:
        try:
            # try a square root transformation
            transformed = np.sqrt(data[feature])
            skewness = stats.skew(transformed.dropna())
            # print(f"Sqrt transform: {feature}, skewness: {skewness}")
        except:
            print("FAILED")
    
    if abs(skewness) < 0.5:
        transformed_data[feature] = transformed
        transformed_feats.append(feature)
    # Plot the transformed histogram
    plt.hist(transformed, bins=50)
    plt.title('Transformed Histogram of {}'.format(feature))
    plt.savefig('plots/Transformed Histogram of {}.png'.format(feature))
    plt.clf()


# Part 3

# Adding TARGET column to the transformed data
transformed_data['TARGET'] = target

for feature in transformed_feats:
    q1 = transformed_data[feature].quantile(0.25)
    q3 = transformed_data[feature].quantile(0.75)
    iqr = q3 - q1
    transformed_data = transformed_data[(transformed_data[feature] >= q1 - 1.5 * iqr) & (transformed_data[feature] <= q3 + 1.5 * iqr)]


for column in transformed_feats:
    plt.figure()
    plt.boxplot(transformed_data[column])
    plt.title(column)
    plt.savefig('plots/Boxplot for {}.png'.format(column))
    plt.clf()


# Generate separate boxplots for each value of the target column
for column in transformed_feats:
    plt.figure()
    plt.boxplot([transformed_data[column][transformed_data['TARGET']==0], transformed_data[column][transformed_data['TARGET']==1]])
    plt.title(column)
    plt.xticks([1, 2], ['TARGET=0', 'TARGET=1'])
    plt.savefig('plots/Boxplot for {} w Target.png'.format(column))
    plt.clf()



# Generate separate boxplots for different levels of education
education_order = ["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education"]
for column in transformed_feats:
    plt.figure()
    plt.boxplot([transformed_data[column][transformed_data['NAME_EDUCATION_TYPE'] == ed] for ed in education_order])
    plt.title(column)
    plt.xticks(range(1, 5), education_order, rotation=45)
    plt.savefig('plots/Boxplot for {} by Education.png'.format(column))
    plt.clf()




# Part 4

# Generate a barplot of housing types
housing_counts = transformed_data['NAME_HOUSING_TYPE'].value_counts()
plt.bar(housing_counts.index, housing_counts.values)
plt.title('Number of Applicants by Housing Type')
plt.xlabel('Housing Type')
plt.ylabel('Number of Applicants')
plt.savefig('plots/Number of Applicants by Housing Type.png')
plt.clf()


# Generate a stacked barplot of housing types by family status
family_housing_counts = transformed_data.groupby('NAME_FAMILY_STATUS')['NAME_HOUSING_TYPE'].value_counts().unstack()
family_housing_counts.plot(kind='bar', stacked=True)
plt.title('Number of Applicants by Housing Type and Family Status')
plt.xlabel('Family Status')
plt.ylabel('Number of Applicants')
plt.legend(title='Housing Type', loc='upper left')
plt.savefig('plots/Number of Applicants by Housing Type and Family Status.png')
plt.clf()



# Part 5



# Creating new column AGE from DAYS_BIRTH by dividing the entries by 365
transformed_data['AGE'] = np.abs(transformed_data['DAYS_BIRTH']) / 365

# Creating new column AGE_GROUP depending on the AGE
bins = [19, 25, 35, 60, np.inf]
labels = ['Very_Young', 'Young', 'Middle_Age', 'Senior_Citizen']
transformed_data['AGE_GROUP'] = pd.cut(transformed_data['AGE'], bins=bins, labels=labels, include_lowest=True)


# Plotting the proportion of applicants with "TARGET"=1 within each age group
grouped = transformed_data.groupby('AGE_GROUP')['TARGET'].mean()
plt.bar(grouped.index, grouped.values)
plt.title('Proportion of Applicants with Target = 1 by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Proportion with Target = 1')
plt.savefig('plots/Proportion of Applicants with Target = 1 by Age Group.png')
plt.clf()


# Plotting the proportion of applicants with "TARGET"=1 within each age group and gender
grouped = transformed_data.groupby(['AGE_GROUP', 'CODE_GENDER'])['TARGET'].mean().unstack()
grouped.plot(kind='bar')
plt.title('Proportion of Applicants with Target = 1 by Age Group and Gender')
plt.xlabel('Age Group')
plt.ylabel('Proportion with Target = 1')
plt.legend(title='Gender', loc='upper left')
plt.savefig('plots/Proportion of Applicants with Target = 1 by Age Group and Gender.png')
plt.clf()
