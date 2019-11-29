# example 1.r: 

# using functions in equivalence package
# Use install.packages('equivalence') to download the package
#   before first use  

library(equivalence)

# calculations from summary statistics (group means and pooled sd)
# Note tost.stat() wants sd, not se

tost.stat(4.16-4.34, 0.19*sqrt(15), 15, Epsilon=0.15*4.34)
# arguments are the difference in group means, the pooled sd,
#   the sample size, and the +/- equivalence region

# using raw data
briskwalk <- read.csv('example 1.csv', as.is=T)
with(briskwalk, tost(criterion, surrogate, epsilon=0.15*4.34, paired=T) )

# Or, doing calculations "by hand", using difference variable in the data set

Y <- briskwalk$difference
nDiff <- sum(!is.na(Y))  	# number of non-missing values
meanDiff <- mean(Y, na.rm=T)	# their average
seDiff <- sd(Y, na.rm=T) / sqrt(nDiff)	# and se

# biologically-motivated equivalence region for change from criterion
lower <- 0.15*4.34
upper <- 0.15*4.34

equivT1 <- (meanDiff + lower)/seDiff
equivT2 <- (meanDiff - upper)/seDiff;

PT1 <- pt(equivT1, nDiff-1, lower=F)
PT2 <- pt(equivT2, nDiff-1)
Pequiv <- max(PT1, PT2)
Pequiv


