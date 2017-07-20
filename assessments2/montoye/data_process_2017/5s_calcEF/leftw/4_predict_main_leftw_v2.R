#setwd('E:/Projects/accelerometer-project/assessments2/montoye/data_process_2017/5s_calcEF/leftw/')

library(nnet)
library(caret)

nnet_wrist<-load("../../models/V2_LW.RData")

files <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/left_wrist/5sec/input_files/')
input_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/left_wrist/5sec/input_files/'
output_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/left_wrist/5sec/v2/result_files/'
for (file  in files){
  EEprediction1<-read.table(file=paste(input_foldername, file, sep=""), header=TRUE)
  pred_Hip <- predict(get(nnet_wrist),EEprediction1)
  temp_name <- paste(output_foldername, paste(strsplit(file, '_FE.txt')[1], "_predicted_left_wrist_PAintensity_2017.csv", sep=""), sep="")
  write.csv(pred_Hip, temp_name)
  print(paste('predicted for left - ', file))
}