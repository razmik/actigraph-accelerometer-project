#setwd('E:/Projects/accelerometer-project/assessments/montoye')
library(nnet)
library(caret)

nnet_Hip<-load("../model/sample_code_2015/AGhipANN_FeatureSet2_2015.RData")

files <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/input_files/')
input_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/input_files/'
output_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/result_files/'
for (file  in files){
  EEprediction1<-read.table(file=paste(input_foldername, file, sep=""), header=TRUE)
  pred_Hip <- predict(get(nnet_Hip),EEprediction1)
  temp_name <- paste(output_foldername, paste(strsplit(file, '_FE.txt')[1], "_predicted_2015.csv", sep=""), sep="") 
  write.csv(pred_Hip, temp_name)
}

#library(nnet)
#library(caret)

#EEprediction1<-read.table(file='sample_code_2015/Participant 1_2 example data_MSSE 2015 study.txt',header=TRUE)
#EEprediction1<-read.table(file='D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/input_files/LSM203 Waist (2016-11-02)_FE.txt',header=TRUE)

#nnet_Hip<-load("sample_code_2015/AGhipANN_FeatureSet2.RData")


#pred_Hip <- predict(get(nnet_Hip),EEprediction1)


#write.csv(pred_Hip, 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/result_files/LSM203 Waist (2016-11-02)_predictions.csv')

