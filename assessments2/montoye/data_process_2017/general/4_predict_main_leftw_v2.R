#setwd('E:/Projects/accelerometer-project/assessments2/montoye/data_process_2017/5s_calcEF/leftw/')

library(nnet)
library(caret)

epochs = c('Epoch5', 'Epoch15', 'Epoch30', 'Epoch60')
wrists = c('left_wrist', 'right_wrist')

print('v2')

for (epoch in epochs){
  for (wrist in wrists){

    if (wrist == 'left_wrist'){
      nnet_wrist <- load("../../models/V2_LW.RData")
    }else if(wrist == 'right_wrist'){
      nnet_wrist <- load("../../models/V2_RW.RData")
    }

    a <- 'E:/Data/Accelerometer_Montoye_ANN/2017/'
    b <- wrist
    c <- '/'
    d <- epoch
    e <- '/input_files/'
    f <- '/v2/result_files/'

    input_foldername <- paste(rbind(a,b,c,d,e), collapse = '')
    files <- list.files(input_foldername)
    output_foldername <- paste(rbind(a,b,c,d,f), collapse = '')

    for (file  in files){
      EEprediction1<-read.table(file=paste(input_foldername, file, sep=""), header=TRUE)
      pred_Hip <- predict(get(nnet_wrist),EEprediction1)
      output_filename <- paste(rbind(output_foldername, strsplit(file, '.txt')[1], '_predicted_', wrist, '_PAintensity_2017.csv'), collapse = '')
      write.csv(pred_Hip, output_filename)
      print(paste('predicted for right - ', output_filename))
    }

  }
}
#
# nnet_wrist<-load("../../models/V2_LW.RData")
#
# files <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/left_wrist/5sec/input_files/')
# input_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/left_wrist/5sec/input_files/'
# output_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Assessment2/Montoye_2017_predictions/left_wrist/5sec/v2/result_files/'
# for (file  in files){
#   EEprediction1<-read.table(file=paste(input_foldername, file, sep=""), header=TRUE)
#   pred_Hip <- predict(get(nnet_wrist),EEprediction1)
#   temp_name <- paste(output_foldername, paste(strsplit(file, '_FE.txt')[1], "_predicted_left_wrist_PAintensity_2017.csv", sep=""), sep="")
#   write.csv(pred_Hip, temp_name)
#   print(paste('predicted for left - ', file))
# }