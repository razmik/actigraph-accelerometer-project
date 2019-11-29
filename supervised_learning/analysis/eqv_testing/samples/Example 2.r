# Example 2.r

# Note: If data are assumed to be log normally distributed,
#  can log transform VO2mile and VO2pacer, 
#  then use tost() in the equivalence library with equiv. region
#    of log(0.98), log(1.0204)

# As of May 2017, need to do ratio equivalence test by specifying
#   the computations

# equivalence region is 0.98 < VO2mile / VO2pacer < 1.0204
# i.e., within 2% of each other.

pacer <- read.csv('Example 2.csv')

# calculate shifted differences and number of non-missing obs.
Da <- with(pacer, VO2mile - 0.98*VO2pacer)
Db <- with(pacer, VO2mile - 1.0204*VO2pacer)
n <- sum(!is.na(Da))

# helper function to calculate the standard error of a mean
se <- function(x) {sd(x, na.rm=T)/sqrt(sum(!is.na(x)))}

# calculate the two T statistics
Ta <- mean(Da)/se(Da)
Tb <- mean(Db)/se(Db)

# and the associated one-sided p-values
Pa <- pt(Ta, n-1, lower=F)
Pb <- pt(Tb, n-1)

# report each p-value and the overal equivalence p-value
c(Pa=Pa, Pb=Pb, Pequiv=max(Pa,Pb))



