# example 1.r: 

# using functions in equivalence package
# Use install.packages('equivalence') to download the package
#   before first use  

library(equivalence)

# calculations from summary statistics (group means and pooled sd)
# Note tost.stat() wants sd, not se
# mean from the meanvalues: 3.00446702 -	3.180920458
percentate = 0.18
tost.stat(3.037839877169385-3.184930938567715, 0.19*sqrt(percentate * 100), 110, Epsilon=percentate*3.037839877169385)
# arguments are the difference in group means, the pooled sd,
#   the sample size, and the +/- equivalence region

# using raw data
briskwalk <- read.csv('hilderbrand.csv', as.is=T)
with(briskwalk, tost(waist_ee_mean, predicted_ee_mean, epsilon=percentate*3.037839877169385, paired=T) )

# Or, doing calculations "by hand", using difference variable in the data set

Y <- briskwalk$realdiff_mean
nDiff <- sum(!is.na(Y))  	# number of non-missing values
meanDiff <- mean(Y, na.rm=T)	# their average
seDiff <- sd(Y, na.rm=T) / sqrt(nDiff)	# and se

# biologically-motivated equivalence region for change from criterion
lower <- percentate*3.037839877169385
upper <- percentate*3.037839877169385

equivT1 <- (meanDiff + lower)/seDiff
equivT2 <- (meanDiff - upper)/seDiff;

PT1 <- pt(equivT1, nDiff-1, lower=F)
PT2 <- pt(equivT2, nDiff-1)
Pequiv <- max(PT1, PT2)
Pequiv