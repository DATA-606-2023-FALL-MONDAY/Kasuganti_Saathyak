## 1. Title and Author

**Project Title:** In-Vehicle Coupon Recommendation System Using Machine Learning

Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang

**Author Name:** Saathyak Rao Kasuganti

**GitHub:** https://github.com/Saathyak

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


**Feature Description:**

- **destination**
  - Data type: Categorical
  - Description: The intended destination of the driver.
  - Possible data entries: "No Urgent Place", "Home", "Work"

- **passenger**
  - Data type: Categorical
  - Description: The type of passengers present in the car.
  - Possible data entries: "Alone", "Friend(s)", "Kid(s)", "Partner"

- **weather**
  - Data type: Categorical
  - Description: The weather conditions at the time of driving.
  - Possible data entries: "Sunny", "Rainy", "Snowy"

- **temperature**
  - Data type: Continuous
  - Description: The temperature in degrees Fahrenheit.
  - Possible data entries: 55, 80, 30 (numerical values)

- **time**
  - Data type: Categorical
  - Description: The time of day when the driving scenario takes place.
  - Possible data entries: "2PM", "10AM", "6PM", "7AM", "10PM"

- **coupon**
  - Data type: Categorical
  - Description: The type of coupon offered.
  - Possible data entries: "Restaurant(<$20)", "Coffee House", "Carry out & Take away", "Bar", "Restaurant($20-$50)"

- **expiration**
  - Data type: Categorical
  - Description: The expiration period of the coupon.
  - Possible data entries: "1d" (1 day), "2h" (2 hours)

- **gender**
  - Data type: Categorical
  - Description: Gender of the driver.
  - Possible data entries: "Female", "Male"

- **age**
  - Data type: Categorical
  - Description: Age group of the driver.
  - Possible data entries: "21", "46", "26", "31", "41", "50plus", "36", "below21"

- **maritalStatus**
  - Data type: Categorical
  - Description: Marital status of the driver.
  - Possible data entries: "Unmarried partner", "Single", "Married partner", "Divorced", "Widowed"

- **has_Children**
  - Data type: Categorical
  - Description: Whether the driver has children (binary).
  - Possible data entries: 1 (Yes), 0 (No)

- **education**
  - Data type: Categorical
  - Description: Educational level of the driver.
  - Possible data entries: "Some college - no degree", "Bachelors degree", "Associates degree", "High School Graduate", "Graduate degree (Masters or Doctorate)", "Some High School"

- **occupation**
  - Data type: Categorical
  - Description: Occupation of the driver.
  - Possible data entries: [List of various occupation types]

- **income**
  - Data type: Categorical
  - Description: Annual income range of the driver.
  - Possible data entries: ["$37500 - $49999", "$62500 - $74999", "$12500 - $24999", "$75000 - $87499", "$50000 - $62499", "$25000 - $37499", "$100000 or More", "$87500 - $99999", "Less than $12500"]

- **Bar**
  - Data type: Categorical
  - Description: Frequency of visiting bars every month.
  - Possible data entries: "never", "less1", "13", "gt8", "nan48"

- **CoffeeHouse**
  - Data type: Categorical
  - Description: Frequency of visiting coffee houses every month.
  - Possible data entries: "never", "less1", "48", "13", "gt8", "nan"

- **CarryAway**
  - Data type: Categorical
  - Description: Frequency of getting take-away food every month.
  - Possible data entries: "n48", "13", "gt8", "less1", "never"

- **RestaurantLessThan20**
  - Data type: Categorical
  - Description: Frequency of going to restaurants with an average expense per person of less than $20 every month.
  - Possible data entries: "48", "13", "less1", "gt8", "never"

- **Restaurant20To50**
  - Data type: Categorical
  - Description: Frequency of going to restaurants with an average expense per person of $20 - $50 every month.
  - Possible data entries: "13", "less1", "never", "gt8", "48", "nan"

- **toCoupon_GEQ15min**
  - Data type: Categorical
  - Description: Whether the driving distance to the restaurant/bar for using the coupon is greater than 15 minutes (binary).
  - Possible data entries: 0 (No), 1 (Yes)

- **toCoupon_GEQ25min**
  - Data type: Categorical
  - Description: Whether the driving distance to the restaurant/bar for using the coupon is greater than 25 minutes (binary).
  - Possible data entries: 0 (No), 1 (Yes)

- **direction_same**
  - Data type: Categorical
  - Description: Whether the restaurant/bar is in the same direction as the current destination (binary).
  - Possible data entries: 0 (No), 1 (Yes)

- **direction_opp**
  - Data type: Categorical
  - Description: Whether the restaurant/bar is in the opposite direction of the current destination (binary).
  - Possible data entries: 1 (Yes), 0 (No)

- **Y**
  - Data type: Categorical
  - Description: Whether the coupon is accepted (binary).
  - Possible data entries: 1 (Yes), 0 (No)
