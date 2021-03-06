#setwd('E:/Projects/accelerometer-project/assessments2/montoye/data_process_2017/5s_calcEF/rightw/')

library(nnet)
library(caret)

nnet_wrist<-load("../../models/V1V2_RW.RData")

files <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/right_wrist/5sec/input_files/')
input_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/right_wrist/5sec/input_files/'
output_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/right_wrist/5sec/v1v2/result_files/'
for (file  in files){
  EEprediction1<-read.table(file=paste(input_foldername, file, sep=""), header=TRUE)
  pred_Hip <- predict(get(nnet_wrist),EEprediction1)
  temp_name <- paste(output_foldername, paste(strsplit(file, '_FE.txt')[1], "_predicted_right_wrist_PAintensity_2017.csv", sep=""), sep="")
  write.csv(pred_Hip, temp_name)
  print(paste('predicted for right - ', file))
}