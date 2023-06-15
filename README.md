# Recipe Time Model
**Authors**: Feiyang Jiang, Yiming Jia
## Project Overview
This is a data science project on investigating the relationship between details of the recipe and the time it takes for completion, in order to construct a model that classifies recipes in terms of time consumption. This project is for DSC80 at UCSD. The dataset used to investigate the topic can be find [here](https://dsc80.com/project3/recipes-and-ratings/food.com), and is originally scrapped from [this source](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf). This project is for DSC80 at UCSD. Our exploratory data analysis on this dataset can be found [here](https://fjiang316.github.io/Recipe_Calorie/).

---
## Investigating Topic and Introduction
Nowadays, with the increase urbanization along with the progress in economy, people lives a faster paced lifestyle than before. With more pressure, there are many indicators that shows people's choice on cooking has been limited in which they couldn't spend as much time trying new recipes as before. This phenomenon among many people indicates the need to clearly know the amount of time a new recipe would require before trying it out to not interrupt their pace of living. Recognizing this need, our project aims to develop a predictive model that identifies the key factors influencing the time required to complete a recipe. By doing so, we empower individuals to make accurate estimations of the time they need to allocate for preparing a new recipe in the future, enabling better time management and planning. **To do so, we will investigate on constructing a multiple-class classifier model on recipes that classifies its time consumption to four different levels (As discussed in next part)**.

### Introduction to the Datasets

The first data set we are using contains the information of 83782 recipes from 2008 to 2018 on food.com, with 10 columns recording the following information:

|Column	                 |Description|
|---                     |---        |
|`'name'	`            |Recipe name|
|`'id'`	                 |Recipe ID|
|`'minutes'`	         |Minutes to prepare recipe|
|`'contributor_id'`	     |User ID who submitted this recipe|
|`'submitted'`	            | Date recipe was submitted|
|`'tags'`	              |Food.com tags for recipe|
|`'nutrition'`	          |Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein    (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value”|
|`'n_steps'`	          |Number of steps in recipe|
|`'steps'`	              |Text for recipe steps, in order|
|`'description'`	     | User-provided description|

The other data set we used contains people's opinions to recipes on food.com, which is consists of 731,927 total reviews. It consists of 5 different columns as listed below:

|Column|Description|
|---|---|
|`'user_id'`	|User ID|
|`'recipe_id'`	|Recipe ID|
|`'date'`	|Date of interaction|
|`'rating'`	|Rating given|
|`'review'`	|Review text|

In our study, we mainly used six columns. We used `minutes`, `nutrition`, `n_steps`, `n_ingredients`, `tags` in the first dataset. Since we only focus on the protein level, we seperated them from the `nutrition` column into separate column and assigned it as `protein` (we will address this column as `protein` for future reference in this study). 

Also, because we would use the `minutes` column as the predicted time consumption for our model, we classified the time in `minutes` into four different levels as shown below:

*Light Meal*  -  less than 30 minutes

*Casual Meal*  -  between 30 minutes to an hour

*Event Meal*  -  between an hour to 4 hours

*Luxury Meal*  -  exceed 4 hours

**We store the categorized minites in a new column named `level`, and this column will be used as our prediction column from now on in this project.**

We also classifies the recipes based on difficulty levels as suggested by specific tags in the `tag` column:

*Easy*  -  if the recipe's tag contains either `easy` or `beginner-cook` that indicates it's on the easier side of all recipes.

*Hard*  -  if the recipe's tag contains any of `for-large-groups`, `dinner-party`, `holiday-event`, we considered it relatively hard.

*Neutral*  -  if the recipe's tag does not include any of the above difficulty indicator tags, we considered it neutral in terms of difficulty.

The resulting difficulty level as extracted from `tags` will be stored in a new column called `difficulty`, and we will address it as this from now on.

We used `date` column in the second dataset, it contains information about when people submitted their opinions on the recipes, which is a relatively useful condition that we could use to access the fairness of our prediction model later on. Since we will be dividing data to groups in terms of years, we extracted year from the `date` column, and assigned this as a new column named `year`.

After data cleaning, the combined DataFrame looks like the following (only showing the first 5 rows for illustration, the actual combined DataFrame has 234425 records):

| level       |   protein | difficulty   |   n_steps |   n_ingredients |   comment_year |
|:------------|----------:|:-------------|----------:|----------------:|---------------:|
| casual meal |         3 | hard         |        10 |               9 |           2008 |
| casual meal |        22 | hard         |        12 |              11 |           2012 |
| casual meal |        32 | easy         |         6 |               9 |           2008 |
| casual meal |        32 | easy         |         6 |               9 |           2009 |
| casual meal |        32 | easy         |         6 |               9 |           2013 |

---
## Framing Investigation Problem
**Investigation Problem**: In this project, we are going to perform an investigation on predicting the time duration to prepare recipes. Specifically, we will make a <u>multiclass classification model</u> in which the model will help classify the recipes into one of the four categories (light, casual, event, luxury) in terms of preparation duration to help people decide on the recipe that fits their pace of living.

**Response Variable**: `level` column that indicates the level of time consumption of a recipe. Because we want a prediction on the time consumption of a recipe categorized in different categories of timeframe, as shown in the data cleanning process of previous part, `level` is a good indicator.

**Measureing Metrics**: the metrics for the performance of our classification model would be <u>*accuracy*</u>, this is because our distribution of level of time consumption in our observed datasets are distributed relatively fairly, so we could use accuracy as a fair metrics that relect the true performance of our model.

**Information At Time of Prediction**: At the time of prediction, the user would normally have the recipe's ingredients, instructions, nutrition table, and the tags where they find the recipe at hand. Through these details on the recipe, people can resonably count the number of steps and ingredients used in the recipe, determine its difficulty based on its tags, and its estimated protein amount in the nutrition table of the recipe. Hence, we would have all four features columns available at the time of prediction.

---
## Baseline Model
**Features Used**:
For the Baseline Model, we decided to use two features as shown below:

- `difficulty`: This feature extracted from the tags indicates the difficulty level of a recipe, and could be used as an useful feature in predicting the time consumption of the recipe. This is because recipes harder to perform tends to take up more time than easier recipes due to more complicated instructions and techniques required. 