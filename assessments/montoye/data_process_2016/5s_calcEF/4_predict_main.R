#setwd('E:/Projects/accelerometer-project/assessments/montoye/data_process_2016/5s_calcEF')
#source('4_predict_main.R')

library(nnet)
library(caret)

nnet_left_wrist<-load("../../models/left_PAintensity_2016.RData")
nnet_right_wrist<-load("../../models/right_PAintensity_2016.RData")

files_left <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2016_predictions/left_wrist/5sec/input_files/')
input_foldername_left <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2016_predictions/left_wrist/5sec/input_files/'
output_foldername_left <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2016_predictions/left_wrist/5sec/result_files/'

files_right <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2016_predictions/right_wrist/5sec/input_files/')
input_foldername_right <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2016_predictions/right_wrist/5sec/input_files/'
output_foldername_right <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2016_predictions/right_wrist/5sec/result_files/'

for (file  in files_left){
  EEprediction1<-read.table(file=paste(input_foldername_left, file, sep=""), header=TRUE)
  pred_Hip <- predict(get(nnet_left_wrist),EEprediction1)
  temp_name <- paste(output_foldername_left, paste(strsplit(file, '_FE.txt')[1], "_predicted_left_wrist_PAintensity_2016.csv", sep=""), sep="")
  write.csv(pred_Hip, temp_name)
  print(paste('predicted left for ', file))
}

for (file  in files_right){
  EEprediction1<-read.table(file=paste(input_foldername_right, file, sep=""), header=TRUE)
  pred_Hip <- predict(get(nnet_right_wrist),EEprediction1)
  temp_name <- paste(output_foldername_right, paste(strsplit(file, '_FE.txt')[1], "_predicted_right_wrist_PAintensity_2016.csv", sep=""), sep="")
  write.csv(pred_Hip, temp_name)
  print(paste('predicted right for ', file))
}