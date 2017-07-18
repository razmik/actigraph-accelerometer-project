files <- list.files('D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/input_files/')
input_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/input_files/'
output_foldername <- 'D:/Accelerometer Data/Montoye/Features/LSM2/Week 1/Wednesday/Montoye_2015_predictions/result_files/'
for (file  in files){
  temp_name <- paste(output_foldername, paste(strsplit(file, '_FE.txt')[1], "_predicted_2015.csv", sep=""), sep="") 
print(temp_name )  
}
