library(equivalence)

percentage = 0.05

input_files <- list.files('output/preprocess/silr_sb/')
input_foldername <- 'output/preprocess/silr_sb/'
output_foldername <- 'output/tost/'

for (file  in input_files) {
input_data <- read.csv(paste(input_foldername, file, sep=""), as.is=T)

freedson_mean = mean(input_data$freedson,na.rm=T)
predicting_mean = mean(input_data$predicting,na.rm=T)
n = nrow(input_data)

# calculations from summary statistics (group means and pooled sd)
# Note tost.stat() wants sd, not se
tost.stat(predicting_mean-freedson_mean, 0.19*sqrt(percentage*100), n, Epsilon=percentage*freedson_mean)
# arguments are the difference in group means, the pooled sd,
#   the sample size, and the +/- equivalence region

result <- with(input_data, tost(freedson, predicting, epsilon=percentage*freedson_mean, paired=T) )

print(file)
print(result)
print("")
print("")
print("")
print("")
  #temp_name <- paste(output_foldername, paste(strsplit(file, '_FE.txt')[1], "_predicted_right_wrist_PAintensity_2017.csv", sep=""), sep="")
  #write.csv(pred_Hip, temp_name)
  #print(paste('predicted for right - ', file))
}