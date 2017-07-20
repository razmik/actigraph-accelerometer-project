#setwd('E:/Projects/accelerometer-project/assessments2/montoye/data_process_2016/60s_calcEF/leftw/')

library(nnet)
library(caret)

nnet_wrist<-load("E:/Projects/accelerometer-project/assessments/montoye/models/left_PAintensity_2016.RData")

files <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2016_predictions/left_wrist/60sec/input_files/')
input_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2016_predictions/left_wrist/60sec/input_files/'
output_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2016_predictions/left_wrist/60sec/result_files/'
for (file  in files){
  EEprediction1<-read.table(file=paste(input_foldername, file, sep=""), header=TRUE)
  pred_Hip <- predict(get(nnet_wrist),EEprediction1)
  temp_name <- paste(output_foldername, paste(strsplit(file, '_FE.txt')[1], "_predicted_left_wrist_PAintensity_2016.csv", sep=""), sep="")
  write.csv(pred_Hip, temp_name)
  print(paste('predicted for right - ', file))
}