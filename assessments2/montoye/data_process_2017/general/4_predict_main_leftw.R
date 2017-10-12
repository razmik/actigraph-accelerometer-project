#setwd('E:/Projects/accelerometer-project/assessments2/montoye/data_process_2017/general/')

library(nnet)
library(caret)

epochs = c('Epoch5', 'Epoch15', 'Epoch30', 'Epoch60')
#wrists = c('left_wrist', 'right_wrist')
wrists = c('left_wrist')

print('v1v2')

for (epoch in epochs){
  for (wrist in wrists){

    if (wrist == 'left_wrist'){
      nnet_wrist<-load("../models/V1V2_LW.RData")
    }else if(wrist == 'right_wrist'){
      nnet_wrist<-load("../models/V1V2_RW.RData")
    }

    a <- 'E:/Data/Accelerometer_Montoye_ANN/2017/'
    b <- wrist
    c <- '/'
    d <- epoch
    e <- '/input_files/'
    f <- '/v1v2/result_files/'

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