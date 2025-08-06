# Bank Haolim Loan Application Analysis

Welcome to the Bank Haolim loan application presentation repository.

## Project Overview

The goal of this project is to analyze how various factors influence loan approval decisions at Bank Haolim, using a loan dataset inspired by Kaggle.

We explore the data, create key performance indicators (KPIs), build predictive models, and provide a deployment solution with an API for loan approval prediction.

**Data source:** Kaggle loan dataset


The main dataset used in this project is "train.csv", which contains information related to loan applications, including demographic data, financial details, and loan approval status.
After performing preprocessing steps — such as handling missing values, encoding categorical features, and removing outliers — a clean version of the dataset was created and saved as df2.

Additional engineered variables were added to df2 to support deeper analysis and modeling. This cleaned and enriched dataset (df2) was exported to a separate file and re-imported during the KPI calculation and data visualization phase.

All the details of the study, including graphs and explanations, are available in the file **Projet_Data_Analyst.pdf**.


---
## Project Outline

1. **Data Understanding : 0_Data_understanding.ipynb** 
   Gain insights into the dataset, variables, and data quality.

2. **Exploratory Data Analysis : 1_EDA.ipynb**  
   Visualize and summarize the data to uncover patterns and relationships.

3. **Data Preprocessing : 2_Preprocessing.ipynb**  
   Clean the data, handle missing values, encode categorical variables, and prepare features for modeling.

4. **Key Performance Indicators : 3_KPI.ipynb**  
   Define and calculate relevant KPIs to evaluate loan applicant profiles and approval trends.

5. **Modeling : 4_Modeling.ipynb**  
   Build machine learning models to predict loan approval status.

6. **Local Flask Application**


#### **Structure**


- **api.py**  
  This is the main file that runs the Flask web application.  
  It contains the code to start the server, manage routes (URLs), handle user requests (like when they submit a form), and send back responses (like displaying a webpage).

- **model.py**  
  This file contains the "business logic" or core functionality of the app.  
  For example, here you might have the model that decides if a loan should be approved or rejected based on the data received.  
  This is where you write rules, calculations, or algorithms to determine the outcome.

- **templates/**  
  This is a special folder that holds the HTML files — the web pages your app shows to users. Flask looks here to find these templates.

  - **form.html**  
    This page shows a form where users can apply for a loan (e.g., entering their name, income, loan amount, etc.).

  - **valid.html**  
    If the loan application is approved, this page is displayed to inform the user that their loan is accepted.

  - **refused.html**  
    If the loan is denied, this page is shown to let the user know their loan was not approved.


## How to Use This Repository

- **Step 1** : Download the raw dataset ***train.csv*** and run scripts ***0_Data_understanding.ipynb***, ***1_EDA.ipynb***, and ***2_Preprocessing.ipynb*** for data exploration and preprocessing.
- **Step 2** : Then, download the cleaned and enriched dataset ***df2.csv*** and run scripts ***3_KPI.ipynb*** and ***4_Modeling.ipynb*** to train and use the predictive model.
- **Step 3** 
1. Run the ***model.py*** file to train and save the model.
2. Run the ***app.py*** file to start the Flask server.
3. Once the API is running, a link will appear in the terminal like this:

Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


