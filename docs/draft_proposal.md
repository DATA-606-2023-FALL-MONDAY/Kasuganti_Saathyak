## 1. Title and Author

**Project Title:** In-Vehicle Coupon Recommendation System Using Machine Learning

Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang

**Author Name:** Saathyak Rao Kasuganti

**GitHub:** 

**LinkedIn:** https://www.linkedin.com/in/saathyak-rao-kasuganti-241440190/

**PowerPoint presentation file:** 

**Link to your YouTube video:** 
    
## 2. Background

**What is the Project about?**

- With the rise of digitization in our era, most businesses heavily rely on personalized marketing strategies, including coupons and discounts, to attract and retain customers.  

- However, the effectiveness of coupon recommendations can vary significantly depending on the context.  

- This project aims to predict whether individuals will accept coupons in different driving scenarios.  

- By using machine learning techniques on UC Irvine in-vehicle coupon recommendation dataset, this project aims to aid businesses optimize their coupon marketing strategies for better customer engagement and successful conversion.  



**What are your research questions?**


Some of the critical research questions that the project seeks to answer are:

  - What are the major factors that influence an individual's decision to accept or reject a coupon in various driving scenarios?  

  
  -	How does general demographic criteria such as gender, age, marital status, presence of children, education, occupation, and income influence and relate to coupon acceptance in different driving scenarios?  
  
  -	What types of coupons (e.g., restaurant, coffee, carry-out, bar) are most likely to be accepted based on different driving scenarios?  


  -	How can businesses/ entrepreneurs customize their marketing strategies based on the identified factors influencing coupon acceptance?  
  

  The answers to these questions can inform marketing strategies, enhance customer engagement, and improve the effectiveness of successful coupon promotions.


## 3. **Where is the data sourced from? Description about the quality of data, credibility and attributes.**  

**Data sources:** - https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation

**Data size:** - 2213 Kb

**Data shape (# of rows and # columns) :** - (12684, 24) i.e. 12684 rows and 24 columns

**Credibility of Source:** The UCI Machine Learning Repository has a longstanding reputation for providing high-quality datasets, making it a credible source for our research. The repository is maintained by the University of California, Irvine, anddata quality standards.

**Size of Data:** The dataset offers a substantial number of instances, totalling 12,684 records. This size provides sufficient data to conduct statistically meaningful analyses and train predictive models effectively.

**Attributes of Data:** The dataset has 23 distinct features. These attributes depict a wide range of information, including demographic details, driving scenario specifics, coupon-related variables, and behavioural characteristics, contributing to a comprehensive analysis.


**Feature Dictionary**

| Feature                  | Data Type     | Description                                         | Possible Data Entries                               |
|--------------------------|---------------|-----------------------------------------------------|------------------------------------------------------|
| destination              | Categorical   | The intended destination of the driver.            | "No Urgent Place", "Home", "Work"                   |
| passenger                | Categorical   | The type of passengers present in the car.         | "Alone", "Friend(s)", "Kid(s)", "Partner"          |
| weather                  | Categorical   | The weather conditions at the time of driving.     | "Sunny", "Rainy", "Snowy"                           |
| temperature              | Continuous    | The temperature in degrees Fahrenheit.             | 55, 80, 30 (numerical values)                       |
| time                     | Categorical   | The time of day when the driving scenario takes place. | "2PM", "10AM", "6PM", "7AM", "10PM"            |
| coupon                   | Categorical   | The type of coupon offered.                        | "Restaurant(<$20)", "Coffee House", "Carry out & Take away", "Bar", "Restaurant($20-$50)" |
| expiration               | Categorical   | The expiration period of the coupon.               | "1d" (1 day), "2h" (2 hours)                       |
| gender                   | Categorical   | Gender of the driver.                              | "Female", "Male"                                   |
| age                      | Categorical   | Age group of the driver.                           | "21", "46", "26", "31", "41", "50plus", "36", "below21" |
| maritalStatus            | Categorical   | Marital status of the driver.                      | "Unmarried partner", "Single", "Married partner", "Divorced", "Widowed" |
| has_Children             | Categorical   | Whether the driver has children (binary).          | 1 (Yes), 0 (No)                                    |
| education                | Categorical   | Educational level of the driver.                   | "Some college - no degree", "Bachelors degree", "Associates degree", "High School Graduate", "Graduate degree (Masters or Doctorate)", "Some High School" |
| occupation               | Categorical   | Occupation of the driver.                          | [List of various occupation types]                 |
| income                   | Categorical   | Annual income range of the driver.                 | ["$37500 - $49999", "$62500 - $74999", "$12500 - $24999", "$75000 - $87499", "$50000 - $62499", "$25000 - $37499", "$100000 or More", "$87500 - $99999", "Less than $12500"] |
| Bar                      | Categorical   | Frequency of visiting bars every month.            | "never", "less1", "13", "gt8", "nan48"             |
| CoffeeHouse              | Categorical   | Frequency of visiting coffee houses every month.   | "never", "less1", "48", "13", "gt8", "nan"         |
| CarryAway                | Categorical   | Frequency of getting take-away food every month.   | "n48", "13", "gt8", "less1", "never"               |
| RestaurantLessThan20     | Categorical   | Frequency of going to restaurants with an average expense per person of less than $20 every month. | "48", "13", "less1", "gt8", "never" |
| Restaurant20To50         | Categorical   | Frequency of going to restaurants with an average expense per person of $20 - $50 every month. | "13", "less1", "never", "gt8", "48", "nan" |
| toCoupon_GEQ15min        | Categorical   | Whether the driving distance to the restaurant/bar for using the coupon is greater than 15 minutes (binary). | 0 (No), 1 (Yes) |
| toCoupon_GEQ25min        | Categorical   | Whether the driving distance to the restaurant/bar for using the coupon is greater than 25 minutes (binary). | 0 (No), 1 (Yes) |
| direction_same           | Categorical   | Whether the restaurant/bar is in the same direction as the current destination (binary). | 0 (No), 1 (Yes) |
| direction_opp            | Categorical   | Whether the restaurant/bar is in the opposite direction of the current destination (binary). | 1 (Yes), 0 (No) |
| Y                        | Categorical   | Whether the coupon is accepted (binary).           | 1 (Yes), 0 (No)                                    |


## 4. Exploratory Data Analysis - A pointwise summary:

1. Out of the 24 columns of the dataset, 6 of them have missing values.

2. The columns with missing values, 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', and 'Restaurant20To50' seem to be important features that improve the performance of the Machine learning model. We can consider deleting the rows with missing values as the number of missing values ranges between 107 and 217 only.

3. The column 'car' has missing values for most entries; therefore, it is best to drop the column to end up with cleaner data.

4. Next, we check the Data-types of the columns and the number of Non-Null Values:
   - About half of the columns have an object datatype, while the remaining are integers. Most columns have no null values.
   - As most of the numerical columns are essentially categorical values, there is not much insight to gain from the statistical summary generated above.
   - Occupation column has the highest possibilities of values with 25 types of entries, followed by income classified into 9 different classes.

5. All of the features (including temperature) are essentially categorical in nature; a correlation matrix would not be of any benefit. Therefore, to understand the distribution of various features into categories, I have used the ProfileReport class from the pandas_profiling module.
   - There are 74 duplicated rows, which is 0.6% of total rows and needs to be dropped to achieve clean data.
   - 'toCoupon_GEQ5min' neither has duplicates nor missing values, but it has only the value '1' for all rows; therefore, it can be dropped.
   - Occupation has a very high number of distinct entries, which greatly increases the dimensionality of the data when using techniques like one-hot-encoding.
   - Therefore, as we are using one-hot encoding that converts the existing categorical features to binary features, we can consider dropping the column 'occupation'.

6. Drop any last remaining rows with NaN values, and cross-check for missing values to verify. Now we end up with a data frame that still has 12k+ rows and no null values.

7. Finally, before starting the pre-processing for the ML models, we use OneHotEncoder for converting the categorical data to be able to use for various machine learning models.
