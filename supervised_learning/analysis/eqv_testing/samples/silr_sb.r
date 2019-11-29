# example 1.r: 

# using functions in equivalence package
# Use install.packages('equivalence') to download the package
#   before first use  

library(equivalence)

# calculations from summary statistics (group means and pooled sd)
# Note tost.stat() wants sd, not se
# mean from the meanvalues: 3.00446702 -	3.180920458
perc = 0.05
tost.stat(76.09469697-79.32484848, 0.19*sqrt(perc*100), 110, Epsilon=perc*76.09469697)
# arguments are the difference in group means, the pooled sd,
#   the sample size, and the +/- equivalence region

# using raw data
briskwalk <- read.csv('silr_sb.csv', as.is=T)
with(briskwalk, tost(freedson, silr, epsilon=perc*76.09469697, paired=T) )

# Or, doing calculations "by hand", using difference variable in the data set

Y <- briskwalk$diff
nDiff <- sum(!is.na(Y))  	# number of non-missing values
meanDiff <- mean(Y, na.rm=T)	# their average
seDiff <- sd(Y, na.rm=T) / sqrt(nDiff)	# and se

# biologically-motivated equivalence region for change from criterion
lower <- perc*76.09469697
upper <- perc*76.09469697

equivT1 <- (meanDiff + lower)/seDiff
equivT2 <- (meanDiff - upper)/seDiff;

PT1 <- pt(equivT1, nDiff-1, lower=F)
PT2 <- pt(equivT2, nDiff-1)
Pequiv <- max(PT1, PT2)
Pequiv