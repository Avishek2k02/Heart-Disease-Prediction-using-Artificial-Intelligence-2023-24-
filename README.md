![1000004017](https://github.com/user-attachments/assets/1d2f0a66-68db-44ad-97ff-ff2569df8b85)
![1000003996](https://github.com/user-attachments/assets/ade3a149-88bf-4e6f-bb47-9cf6ecc60f98)
![1000004032](https://github.com/user-attachments/assets/71ad0e75-c70a-405b-982c-348578c66d26)
![1000004002](https://github.com/user-attachments/assets/173c56dc-64e9-4141-a5c2-f4a6a75de1f0)
![1000003999](https://github.com/user-attachments/assets/509182ad-cbc7-4bc8-b99c-e8ab9aece066)
![1000004005](https://github.com/user-attachments/assets/b9a46f9a-60ce-450a-9ca4-72fd48b05e4d)
![1000004011](https://github.com/user-attachments/assets/106620e1-e6da-446b-9d4a-a04530a66358)
![1000004008](https://github.com/user-attachments/assets/76c92d24-306d-43e2-8af1-0c23a37a535d)
![1000004023](https://github.com/user-attachments/assets/0d4ad818-49fa-413c-811f-cb0c9b921b5b)
![1000004029](https://github.com/user-attachments/assets/54caf8e3-dd30-482a-895c-eb44470f8347)
![1000004026](https://github.com/user-attachments/assets/7fc35803-811b-44b1-9153-c271182f0ad0)
![1000004014](https://github.com/user-attachments/assets/913588e2-b1a8-4683-bab6-6632f147aae4)
![1000004035](https://github.com/user-attachments/assets/17387f80-8474-49f2-b5fe-51b148c89944)
![1000004021](https://github.com/user-attachments/assets/a5aede59-bcff-4607-a54f-fcbb21602a55)
![1000004019](https://github.com/user-attachments/assets/4bcd6528-d2a2-463a-8938-1c19f147e060)
# Heart-and-Disease-Prediction
Using machine learning for disease prediction involves teaching computers to study lots of medical information to guess if someone might get sick. For example, with heart disease prediction using machine learning, computers can look at factors like age, blood pressure, and cholesterol levels to guess who might have heart problems in the future. This helps doctors catch issues early and keep people healthy.

Importing Necessary Libraries Data Loading Plotting Librariesimport pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns import cufflinks as cf %matplotlib inline

Metrics for Classification techniquefrom sklearn.metrics import classification_report,confusion_matrix,accuracy_score

Scalerfrom sklearn.preprocessing import StandardScaler from sklearn.model_selection import RandomizedSearchCV, train_test_split

Model buildingfrom xgboost import XGBClassifier from catboost import CatBoostClassifier from sklearn.ensemble import RandomForestClassifier from sklearn.neighbors import KNeighborsClassifier from sklearn.svm import SVC

Data Loading Here we will be using the pandas read_csv function to read the dataset. Specify the location of the dataset and import them.

Importing Datadata = pd.read_csv(“heart.csv”) data.head(6) # Mention no of rows to be displayed from the top in the argument

Output:

Exploratory Data Analysis Now, let’s see the size of the datasetdata.shape

Output:(303, 14)

Inference: We have a dataset with 303 rows which indicates a smaller set of data.

As above we saw the size of our dataset now let’s see the type of each feature that our dataset holds.

Python Code:

Inference: The inference we can derive from the above output is:

Out of 14 features, we have 13 int types and only one with the float data types. Woah! Fortunately, this dataset doesn’t hold any missing values. As we are getting some information from each feature so let’s see how statistically the dataset is spread.data.describe()

Output:

Exploratory Data Analysis It is always better to check the correlation between the features so that we can analyze that which feature is negatively correlated and which is positively correlated so, Let’s check the correlation between various features.plt.figure(figsize=(20,12)) sns.set_context(‘notebook’,font_scale = 1.3) sns.heatmap(data.corr(),annot=True,linewidth =2) plt.tight_layout()

Output:

output , heart disease prediction using Machine learning By far we have checked the correlation between the features but it is also a good practice to check the correlation of the target variable.

So, let’s do this!sns.set_context(‘notebook’,font_scale = 2.3) data.drop(‘target’, axis=1).corrwith(data.target).plot(kind=’bar’, grid=True, figsize=(20, 10), title=”Correlation with the target feature”) plt.tight_layout()

Output:

Correlation with the Target Feature , Inference: Insights from the above graph are:

Four feature( “cp”, “restecg”, “thalach”, “slope” ) are positively correlated with the target feature. Other features are negatively correlated with the target feature. So, we have done enough collective analysis now let’s go for the analysis of the individual features which comprises both univariate and bivariate analysis.

Age(“age”) Analysis Here we will be checking the 10 ages and their counts.plt.figure(figsize=(25,12)) sns.set_context(‘notebook’,font_scale = 1.5) sns.barplot(x=data.age.value_counts()[:10].index,y=data.age.value_counts()[:10].values) plt.tight_layout()

Output:

Age Analysis| Heart Disease Prediction Inference: Here we can see that the 58 age column has the highest frequency.

Let’s check the range of age in the dataset.minAge=min(data.age) maxAge=max(data.age) meanAge=data.age.mean() print(‘Min Age :’,minAge) print(‘Max Age :’,maxAge) print(‘Mean Age :’,meanAge)

Output:

Output | Heart Disease Prediction Min Age : 29 Max Age : 77 Mean Age : 54.366336633663366

We should divide the Age feature into three parts – “Young”, “Middle” and “Elder”Young = data[(data.age>=29)&(data.age<40)] Middle = data[(data.age>=40)&(data.age<55)] Elder = data[(data.age>55)] plt.figure(figsize=(23,10)) sns.set_context(‘notebook’,font_scale = 1.5) sns.barplot(x=[‘young ages’,’middle ages’,’elderly ages’],y=[len(Young),len(Middle),len(Elder)]) plt.tight_layout()

Output:

Heart Disease Prediction Inference: Here we can see that elder people are the most affected by heart disease and young ones are the least affected.

To prove the above inference we will plot the pie chart.colors = [‘blue’,’green’,’yellow’] explode = [0,0,0.1] plt.figure(figsize=(10,10)) sns.set_context(‘notebook’,font_scale = 1.2) plt.pie([len(Young),len(Middle),len(Elder)],labels=[‘young ages’,’middle ages’,’elderly ages’],explode=explode,colors=colors, autopct=’%1.1f%%’) plt.tight_layout()

Output:

Sex(“sex”) Feature Analysis Sex feature analysis | Heart Disease Prediction plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘sex’]) plt.tight_layout()

Output:

Inference: Here it is clearly visible that, Ratio of Male to Female is approx 2:1.

Now let’s plot the relation between sex and slope.plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘sex’],hue=data[“slope”]) plt.tight_layout()

Output:

Output of Sex Analysis, Inference: Here it is clearly visible that the slope value is higher in the case of males(1).

Chest Pain Type(“cp”) Analysis plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘cp’]) plt.tight_layout()

Output:

Chest Pain Inference: As seen, there are 4 types of chest pain

status at least condition slightly distressed condition medium problem condition too bad Analyzing cp vs target column

Heart Disease Prediction Inference: From the above graph we can make some inferences,

People having the least chest pain are not likely to have heart disease. People having severe chest pain are likely to have heart disease. Elderly people are more likely to have chest pain.

Thal Analysis plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘thal’]) plt.tight_layout()

Output:

Thal Analysis Target plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘target’]) plt.tight_layout()

Output:

Target | Heart Disease Prediction Inference: The ratio between 1 and 0 is much less than 1.5 which indicates that the target feature is not imbalanced. So for a balanced dataset, we can use accuracy_score as evaluation metrics for our model.

Feature Engineering Now we will see the complete description of the continuous data as well as the categorical datacategorical_val = [] continous_val = [] for column in data.columns: print(“——————–“) print(f”{column} : {data[column].unique()}”) if len(data[column].unique()) <= 10: categorical_val.append(column) else: continous_val.append(column)

Output:

Feature Engineering Output | Heart Disease Prediction Now here first we will be removing the target column from our set of features then we will categorize all the categorical variables using the get dummies method which will create a separate column for each category suppose X variable contains 2 types of unique values then it will create 2 different columns for the X variable.categorical_val.remove(‘target’) dfs = pd.get_dummies(data, columns = categorical_val) dfs.head(6)

Output:

Output | Heart Disease Prediction Now we will be using the standard scaler method to scale down the data so that it won’t raise the outliers also dataset which is scaled to general units leads to having better accuracy.sc = StandardScaler() col_to_scale = [‘age’, ‘trestbps’, ‘chol’, ‘thalach’, ‘oldpeak’] dfs[col_to_scale] = sc.fit_transform(dfs[col_to_scale]) dfs.head(6)

Output:

Output | Heart Disease Prediction Modeling Splitting our DatasetX = dfs.drop(‘target’, axis=1) y = dfs.target X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

The KNN Machine Learning Algorithm knn = KNeighborsClassifier(n_neighbors = 10) knn.fit(X_train,y_train) y_pred1 = knn.predict(X_test) print(accuracy_score(y_test,y_pred1))

Output:0.8571428571428571

Conclusion on Heart Disease Prediction

We did data visualization and data analysis of the target variable, age features, and whatnot along with its univariate analysis and bivariate analysis.

We also did a complete feature engineering part in this article which summons all the valid steps needed for further steps i.e. model building.

From the above model accuracy, KNN is giving us the accuracy which is 89%. Conclusion Heart disease prediction using machine learning utilizes algorithms to analyze medical data like age, blood pressure, and cholesterol levels, aiding in early detection and prevention. Machine learning greatly enhances disease prediction by analyzing large datasets, identifying patterns, and making accurate forecasts, ultimately improving healthcare outcomes and saving lives.
