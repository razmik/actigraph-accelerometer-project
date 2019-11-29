# Example 3.r
# this code assumes no missing values.  
#   if there are missing values, will need to modify the code
#   or pass the data set through na.omit() before doing the computations

allMarkers <- read.csv('Example 3.csv')

all.lm <- lm(surrogate ~ criterion, data=allMarkers)
# Robinson's original regression proposal

# confidence interval method is easy: just get 90% CI's
confint(all.lm, level=0.90)

# calculate variables to fit adjusted regression
allmean <- mean(allMarkers$criterion)
allMarkers$cSurrogate <- allMarkers$surrogate - allmean
allMarkers$cCriterion <- allMarkers$criterion - allmean

all.lm2 <- lm(cSurrogate ~ cCriterion, data=allMarkers)
# adjusted regression, so intercept is average surrogate
#  at average criterion

# confidence interval method is easy: just get 90% CI's
confint(all.lm2, level=0.90)

# compute two-one-sided-tests by writing out the computations 
 
# extract sample size, estimates and their se's from the 
#   regression fit
# illustrated with the adjusted regression
# since all quantities are vectors of length 2, can do 
#   computations for intercept and slope at the same time

n <- dim(allMarkers)[1]  
  # here's one place where assume no missing values
ests <- coef(all.lm2)
ses <- sqrt(diag(vcov(all.lm2)))
# se's of the regression param are sqrt Variance 

lower <- c(0.1*allmean, 0.9)
upper <- c(0.1*allmean, -1.1)
# lower and upper bounds for intercept and slope

Ta <- (ests - lower)/ses
Tb <- (ests + upper)/ses

Pa <- pt(Ta, n-2, lower=F)
Pb <- pt(Tb, n-2)

pmax(Pa, Pb)
# Equivalence p-values for intercept and slope separately

c(Pequiv=max(Pa, Pb))
# overall equivalence p-value = largest of the four component p-values

# R does not provide an easy way to fit a meta regression

