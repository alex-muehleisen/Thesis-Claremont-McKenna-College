# The following R packages were used in analysis:

library(tidyverse)
library(tidyr)
library(ggplot2)
library(dplyr)
library(knitr)
library(readr)
library(reticulate)
library(magrittr)
library(randomForest)
library(datasets)
library(caret)

# Some analysis will be performed in Python, which is possible in R through the *reticulate* package. 
# Therefore, the following libraries and modules will also need to be installed to run the appropriate regressions and models. 
# They are installed using the `py_install` function from the *reticulate* package. 
# As they are already installed on this OS, they are commented out, but still included in this paper for reference. 

#py_install("pandas")
#py_install("scikit-learn")
#py_install("spicy")
#py_install("seaborn")
#py_install("matplotlib")

## Please observe the accompanying Python script for more analysis.

##### Trends in the Data

### Read in School Closure Data

school_closure_data <- read.csv("duration_school_closures.csv")

school_closure_data %>% 
  arrange(desc(Duration.of.FULL.and.PARTIAL.school.closures..in.weeks.)) %>% 
  head(25)

school_closure_data %>% 
  arrange(desc(Duration.of.FULL.and.PARTIAL.school.closures..in.weeks.)) %>% 
  head(20) %>%
  ggplot(aes(ISO, Duration.of.FULL.and.PARTIAL.school.closures..in.weeks.)) +
  geom_col(show.legend = TRUE,  fill='red') +
  theme_minimal() +
  labs(x = "Country Abbreviation", 
       y = "Duration of Full and Partial School Closures (weeks)", 
       title = "School Closures by Country")


### Read in Numeracy and Literature Scores Data

numlit21_data <- read.csv("NumLit_Scores_2021.csv")
numlit19_data <- read.csv("NumLit_Scores_2019.csv")
numlit18_data <- read.csv("NumLit_Scores_2018.csv")
numlit17_data <- read.csv("NumLit_Scores_2017.csv")
numlit16_data <- read.csv("NumLit_Scores_2016.csv")

# The datasets are joined together using a full_join, and the joined dataset is named joined_data. 
# The column that is most interesting from this dataset is the **Mean Scale Score** variable, 
# which contains aggregate values of student scores in Numeracy and Literacy measured tests. 
# We want to do some data transformations with this column, so it needs to be converted to a double. 
# After doing so, all the missing values in the data are changed to NA, which will help manipulate and filter the data. 
# This concludes the data cleaning process for this dataset. 

#### Transforming and Joining the Data

joined_data <- full_join(numlit21_data, numlit19_data)
joined_data <- full_join(joined_data, numlit18_data)
joined_data <- full_join(joined_data, numlit17_data)
#joined_data <- full_join(joined_data, numlit16_data)

# If any of the joins raise an error, as `numlit16data` did here, simply comment out the line and check the data for inconsistencies. 
# In this case, some of the column data types in `numlit16data` did not match the corresponding column data types in `joined_data`. 
# This issue can be resolved by transforming the data so that each column data type is the same in each dataset. 
# The type of the data can be found with the `sapply` function in R. 

numlit16_data$Students.Tested <- as.factor(numlit16_data$Students.Tested)
numlit16_data$Students.with.Scores <- as.factor(numlit16_data$Students.with.Scores)
numlit16_data$Total.Tested.At.Entity.Level <- as.factor(numlit16_data$Total.Tested.At.Entity.Level)
numlit16_data$Total.Tested.with.Scores <- as.factor(numlit16_data$Total.Tested.with.Scores)
numlit16_data$CAASPP.Reported.Enrollment <- as.factor(numlit16_data$CAASPP.Reported.Enrollment)

# After making these changes, the `numlit16data` can be successfully joined with the other datasets as follows:

joined_data <- full_join(joined_data, numlit16_data)

# The last thing that needs to be done to prepare the data for analysis is simply making sure the data type 
# of the target variable can be used with aggregate functions in R. 
# Additionally, changing the data type of the column to *double* forces R to convert any missing values to NA. 
# This is advantageous in terms of the filtering and sorting that will be done later.

joined_data$Mean.Scale.Score <- as.double(joined_data$Mean.Scale.Score)

# The `NAs introduced by coercion` warning is displayed after running this command, 
# which confirms that R has converted missing values in the data to NAs. 

###### Plotting Student Performance

# To look at how COVID-19 affected student performance, `NA` values are first filtered out of 
# the target variable, **Mean Scale Score**, before plotting that Score versus Testing Year. 
# This is achieved using the `geom_smooth` function from the *ggplot2* library. 

joined_data %>% 
  filter(!is.na(Mean.Scale.Score)) %>% 
  group_by(Test.Year, Grade) %>% 
  summarise(Average.Score = mean(Mean.Scale.Score)) %>% 
  ggplot(aes(x = Test.Year, y = Average.Score)) + 
  geom_smooth(se = FALSE) + 
  labs(x = "Test Year", 
       y = "Average Score", 
       title = "Average Student Assessment Scores by Year")

### with Confidence Interval around Smooth
joined_data %>% 
  filter(!is.na(Mean.Scale.Score)) %>% 
  group_by(Test.Year, Grade) %>% 
  summarise(Average.Score = mean(Mean.Scale.Score)) %>% 
  ggplot(aes(x = Test.Year, y = Average.Score)) + 
  geom_smooth() + 
  labs(x = "Test Year", 
       y = "Average Score", 
       title = "Average Student Assessment Scores by Year")

# This confirms the drop-off in results from 2019 to 2021, but the confidence interval on this figure is very broad, 
# so findings should be taken with a grain of salt. 
# Nevertheless, the reason for the drop-off can be inferred to be due to the effect of school closures 
# and the potential build-up of learning-loss over this schooling interruption.  

# To dive further into this, a good place to start might be to split up students by grade. 
# This will determine whether the drop in performance is common across all schooling grades or not. 

plt_data1 <- joined_data %>% 
  filter(!is.na(Mean.Scale.Score)) %>% 
  group_by(Test.Year, Grade) %>% 
  summarise(Average.Score = mean(Mean.Scale.Score))
head(plt_data1, 10)

plt_data1 %>%
  ggplot(aes(x = Test.Year, y = Average.Score)) +
  facet_wrap(vars(Grade), ncol = 3) +
  geom_point() +
  geom_line() +
  labs(x = "Testing Year", 
       y = "Average Score", 
       title = "Average Testing Score by Year (Wrapped by Grade)")

# It looks like 3rd graders had the largest drop-off in performance, followed by 4th, 5th, and 6th graders. 
# Inspecting the plots more closely, this COVID-related drop-off in performance from 2019 to 2021 seems to decrease as students get older. 
# To look into this further, the difference between scores in 2021 and 2019 can be measured and the difference between them can be plotted.

head(plt_data1, 10)

plt_data2 <- plt_data1 %>% 
  filter(Test.Year == 2019) %>% 
  mutate(`2019 Test Scores` = Average.Score)
plt_data3 <- plt_data1 %>% 
  filter(Test.Year == 2021) %>% 
  mutate(`2021 Test Scores` = Average.Score)

var1 <- 1:7
plt_data4 <- tibble(
  `2019 Test Scores` = var1,
  `2021 Test Scores` = var1
)

plt_data4$`2019 Test Scores` <- paste(plt_data2$`2019 Test Scores`)
plt_data4$`2021 Test Scores` <- paste(plt_data3$`2021 Test Scores`)

plt_data4$`Covid Difference` <- as.numeric(plt_data4$`2021 Test Scores`) - as.numeric(plt_data4$`2019 Test Scores`)
plt_data4$`Grade` <- c(3, 4, 5, 6, 7, 8, 11)
plt_data4$Grade <- as.factor(plt_data4$Grade)
plt_data4 <- select(plt_data4, Grade, `2019 Test Scores`, `2021 Test Scores`, `Covid Difference`)
plt_data4

plt_data4 %>% 
  ggplot(aes(Grade, `Covid Difference`)) + 
  geom_point() + 
  geom_col(fill = 'darkblue') + 
  geom_abline(color = 'red', size = 2) + 
  theme_minimal() + 
  labs(title = "Difference in Student Assessment Scores from 2019 to 2021")

# After some more data manipulation, the difference in scores from 2019 to 2021 can be plotted against students’ grade levels. 
# As speculated, the older a student is, the less affected they seem to be from the quarantine schooling interruption. 
# This may be because older students are more independent and are therefore also more likely to be independent learners. 
# Older students are also more mature and more likely to have to study or work towards future professional goals. 
# In their time away from school, students in grade 11 would most likely have studied a lot more than 3rd graders, 
# so their accumulated learning-loss might be a lot less than that of younger students. 

##### Online Courses and Distance Learning

### Read-in Education Digest Data 

distance_learning_numbers <- read.csv("distance_learning_numbers.csv")

distance_learning_numbers %>% 
  ggplot(aes(x=`ï..year`, y=Exclusive.distance)) + 
  geom_bar(position="stack", stat="identity", fill='lightblue') + 
  theme_minimal() +
  labs(title = "COVID-19 Impact on Distance Learning (Exclusive Distance only)")

distance_learning_numbers %>% 
  ggplot(aes(x=`ï..year`, y=Some.distance)) + 
  geom_bar(position="stack", stat="identity", fill = 'orange') + 
  theme_minimal() + 
  labs(title = "COVID-19 Impact on Distance Learning (Some Distance only)")

distance_learning_data <- distance_learning_numbers %>% 
  pivot_longer(!`ï..year`, names_to = "Type of Distance", values_to = "Percentage")

names(distance_learning_data)[names(distance_learning_data)=="ï..year"] <- "Year"
distance_learning_data

distance_learning_data %>% 
  ggplot(aes(x=Year, y=Percentage, fill = `Type of Distance`)) + 
  geom_bar(stat="identity") + 
  theme_minimal() + 
  labs(title = "COVID-19 Impact on Distance Learning")

##### Distance Learning Performance Numbers

### Read-in Open University Data

assessments <- read_csv("assessments.csv")
courses <- read_csv("courses.csv")
studentAssessment <- read_csv("studentAssessment.csv")
studentInfo <- read_csv("studentInfo.csv")
studentRegistration <- read_csv("studentRegistration.csv")
studentVle <- read_csv("studentVle.csv")
vle <- read_csv("vle.csv")

## Data Cleaning

# Looking at the tables, *studentAssessment* seems the most promising, 
# as it includes the `id_assessment` and `id_student` keys, as well as a `score` column. 
# The keys will be especially helpful for joins, while the `score` column may prove to be an 
# excellent target variable to evaluate student performance with. 

head(studentAssessment, 10)

# Having data for the type of assessment might be helpful, 
# so joining the *studentAssessment* table with the *assessments* table is a good idea. 
# This can be accomplished with an `inner_join` from the *dplyr* package, 
# which will join the tables only on matching values in both tables on the `id_assessment` key. 

oulad_data <- studentAssessment %>% inner_join(assessments, by="id_assessment")

# This process can be repeated with the *studentInfo* table, 
# this time attempting to incorporate information on the students into the joined table. 

oulad_data <- oulad_data %>% inner_join(studentInfo, by="id_student")

# The *studentVLE* table contains information on the number of interactions each student made with an online module, 
# but only contains an observation for the number of clicks in a given day. 
# These values can be summed by grouping the data on `id_student` 
# and then adding together the total clicks on a given day for each student. 
# This is accomplished with the the `sum` function from R.

studentVle2 <- studentVle %>% group_by(id_student) %>% summarize(sum_click = sum(sum_click))

# The *studentVLE* table can now be joined with the rest of our data.

oulad_data <- oulad_data %>% inner_join(studentVle2, by="id_student")

oulad_data <- oulad_data %>% select(-date_submitted, -is_banked, 
                                    -code_module.x, -code_presentation.x, 
                                    -code_module.y, -code_presentation.y, 
                                    -date, -gender, -region, -age_band, 
                                    -disability, -num_of_prev_attempts, 
                                    -imd_band)

##### Random Forest Model

# Random Forest is an example of a supervised machine learning algorithm, 
# which means that it is trained using labeled data and a simple method for computational complexity. 
# It quite a common algorithm because of its high accuracy and pertinence to both regression and classification problems. 
# Since the dataset being used here provides information on the final result of each module for each student, 
# placed into three categories (pass, fail, distinction), a Random Forest Model of classification seems like a good option here. 
# This model combines the output of multiple decision trees into a single result, 
# meaning several models are actually created and tested, before a single, final numerical is given. 

# To prepare to build the model, the first thing to do is make sure that our target column is a factor, 
# so that it can be classified correctly.

oulad_data$final_result <- as.factor(oulad_data$final_result)

# Next, the data should be split into training and testing sets, 
# which can be done with the `sample` function. 
# The testing data will represent 30% of the overall data in this case.

set.seed(223)
ind <- sample(2, nrow(oulad_data), replace = TRUE, prob = c(0.7, 0.3))
oulad_train <- oulad_data[ind==1,]
oulad_test <- oulad_data[ind==2,]

# Checking the data for missing values reveals that quite a few values are missing.

sum(!complete.cases(oulad_train))
sum(!complete.cases(oulad_train$score))

# The Random Forest model can be built using the `randomForest` function from the *randomForest* package in R. 
# The number of trees is set to 500 and the 'importance' parameter is set to TRUE. 
# As noted above, there a quite a few missing values in the dataset. 
# The last parameter, 'na.action' is set to 'omit', in order to remove these straggling NAs from the data. 

rf_model <- randomForest(final_result ~ weight+studied_credits+sum_click, 
                         data=oulad_train, 
                         importance=TRUE, 
                         ntree = 500, 
                         na.action = na.omit)

# Next, the model is used to predict training values. 
# A confusion matrix can be made to observe the accuracy of the model. 

p2_train <- predict(rf_model, oulad_train)
confusionMatrix(p2_train, oulad_train$final_result)

# Finally, the model is used to predict testing values. 
# A confusion matrix can again be made to observe the accuracy of the model.

p2_test <- predict(rf_model, oulad_test)
confusionMatrix(p2_test, oulad_test$final_result)

# Plotting the Random Forest Model illustrates error as a function of the number of trees in the model.

plot(rf_model)

### Variable Importance

# The importance of feature variables used in the model can be 
# calculated using the `importance` function from the *randomForest* library. 

importance(rf_model)

# These values can be visualized by plotting them with the `varImpPlot` command.

varImpPlot(rf_model)

# The plot reveals that the number of clicks a student makes on a module has the 
# highest level of importance in regard to creating the model on the data. 
# This indicates that the column is the most valuable predictor of the dependent variable. 



