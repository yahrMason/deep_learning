
# Dependencies ------------------------------------------------------------

library(tidyverse)
library(janitor)
library(plyr)


mn = function(x){as.numeric(as.character(x))}


# Data Ingestion and Preprocessing ----------------------------------------

# set directory
setwd('~/house_prices/')

set.seed(100)

# read in data
train = read.csv('train.csv') %>% 
  dplyr::mutate(set = 'train') %>% 
  janitor::clean_names()

test = read.csv('test.csv') %>% 
  dplyr::mutate(set = 'test', SalePrice = NA) %>% 
  janitor::clean_names()

# isolate test set house ids
submission_id = test$id
test = dplyr::select(test, -id)
train = dplyr::select(train, -id)

# function to quicky inspect missing column values
view_na = function(df, name){
  col = df[,which(names(df) == name)]  
  row = which(is.na(col))
  View(df[row, ])
}



# Feature Engineering -----------------------------------------------------
df = rbind.fill(train, test) %>%
  # create house age variables
  dplyr::mutate(
    age = yr_sold - year_built,
    is_remod = !is.na(year_remod_add),
    remod_age = yr_sold - year_remod_add,
    garage_age = dplyr::case_when(
      !is.na(yr_sold) ~ mn(yr_sold) - mn(garage_yr_blt),
      is.na(yr_sold) ~ 2010 - mn(garage_yr_blt)
    )
  ) %>%
  dplyr::select(-c(yr_sold, year_remod_add, year_built, garage_yr_blt)) %>%
  dplyr::mutate_if(is.integer, mn)

train = dplyr::filter(df, set == 'train') %>% 
  dplyr::select(- set) 

test = dplyr::filter(df, set == 'test') %>% 
  dplyr::select(-set)


# Scaling and Treatment ---------------------------------------------------

## Normalize numeric data
num_cols = names(train %>% dplyr::select_if(is.numeric))
trn_num_index = which(names(train) %in% num_cols)
temp = scale(train[,trn_num_index])

train[,trn_num_index] = temp

# get the means and standard deviations from temp, to scale test too
means = attr(temp, "scaled:center")
standard_deviations = attr(temp, "scaled:scale")

# scale test
test_num_index = which(names(test) %in% num_cols)
test[, test_num_index] <- scale(test[, test_num_index], center = means, scale = standard_deviations)


# Treat missing vars
library(vtreat)

# design treatment plan with calibration set
cal_rows = sample(1:nrow(train), size = floor(0.5*nrow(train)), replace = FALSE)
cal = train[cal_rows,]
train = train[-cal_rows,]

treatments = designTreatmentsN(cal, names(cal), 'sale_price')

# prepare the training and testing set
train = prepare(treatments, rbind(train, cal), scale = TRUE)
test = prepare(treatments, test)

# split 
Y = train$sale_price
x = dplyr::select(train, - sale_price)

submission_x = dplyr::select(test, -sale_price)

# remove unneccesary data
rm(test, train, cal, treatments, df, cal_rows, test_num_index, trn_num_index)

# Build Model -------------------------------------------------------------

# Model Description: Deep Neural Network built with Keras
# set up deep learning environment
library(reticulate)
library(keras)
library(tensorflow)

# need to split the train into its own train and test set
split_sample = sample(length(Y), floor(0.75*length(Y)))

train_x = x[split_sample,]
test_x = x[-split_sample,]

train_y = Y[split_sample]
test_y = Y[-split_sample]

# reshpae the x dataframes
train_x = array_reshape(as.matrix(train_x), c(nrow(train_x), ncol(train_x)))
test_x = array_reshape(as.matrix(test_x), c(nrow(test_x), ncol(test_x)))



# Build Model Architecture
model = keras_model_sequential()
model %>%
  layer_dense(units = 150, activation = 'relu', input_shape = c(307)) %>%
  layer_dropout(rate = 0.15) %>%
  layer_dense(units = 75, activation = 'relu') %>%
  layer_dropout(rate = 0.15) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.15) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'linear')
summary(model)
  
# Compile Model
model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(), 
  metrics = c('mse', 'mae')
)
  
# Train Model
history <- model %>% fit(
  train_x, train_y, 
  epochs = 50, batch_size = 75, 
  validation_split = 0.2
)
plot(history)

# assess accuracy on testing set
model %>% evaluate(test_x, test_y)
  
  
# predict on submission 
submission_x = keras::array_reshape(as.matrix(submission_x), c(nrow(submission_x), ncol(submission_x)))
predictions = model %>% predict(submission_x)

# scale the predictions
sp_mn = means[which(names(means) == 'sale_price')]
sp_sd = standard_deviations[which(names(standard_deviations) == 'sale_price')]
predictions_scale = predictions*sp_sd + sp_mn

dat = cbind(submission_id, predictions_scale) %>%
  dplyr::as_data_frame() %>%
  purrr::set_names(c('Id', 'SalePrice'))
write.csv(dat, 'submission.csv',row.names=FALSE)
  
  



