#setwd('E:/Projects/accelerometer-project/assessments2/montoye/data_process_2016/5s_calcEF/leftw/')

library(nnet)
library(caret)

nnet_wrist<-load("E:/Projects/accelerometer-project/assessments/montoye/models/left_PAintensity_2016.RData")

epochs = c('Epoch5', 'Epoch15', 'Epoch30', 'Epoch60')
wrists = c('left_wrist', 'right_wrist')

for (epoch in epochs){
  for (wrist in wrists){

    if (wrist == 'left_wrist'){
      nnet_wrist<-load("E:/Projects/accelerometer-project/assessments/montoye/models/left_PAintensity_2016.RData")
    }else if(wrist == 'right_wrist'){
      nnet_wrist<-load("E:/Projects/accelerometer-project/assessments/montoye/models/right_PAintensity_2016.RData")
    }

    a <- 'E:/Data/Accelerometer_Montoye_ANN/2016/'
    b <- wrist
    c <- '/'
    d <- epoch
    e <- '/input_files/'
    f <- '/result_files/'

    input_foldername <- paste(rbind(a,b,c,d,e), collapse = '')
    files <- list.files(input_foldername)
    output_foldername <- paste(rbind(a,b,c,d,f), collapse = '')

    for (file  in files){
        # if (file == 'LSM112_(2016-10-05)_row_33012_to_37629.txt'){
          EEprediction1<-read.table(file=paste(input_foldername, file, sep=""), header=TRUE)
          pred_Hip <- predict(get(nnet_wrist),EEprediction1)
          output_filename <- paste(rbind(output_foldername, strsplit(file, '.txt')[1], '_predicted_', wrist, '_PAintensity_2016.csv'), collapse = '')
          write.csv(pred_Hip, output_filename)
          print(paste('predicted for right - ', output_filename))
        # }
    }

  }
}