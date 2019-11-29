ods html close; ods html;

proc import datafile='Example 3.csv' out=overallMean replace;
run;

/* Need to calculate the criterion mean over all activities */
/*  and merge that into the data set */
/* Here is one way.  There are others  */

/* Use proc standard to center the criterion.  The difference between
     that and the original value is the overall mean */
      
proc standard data=overallmean mean=0
    out=center(rename=(criterion=center) drop=surrogate);
  var criterion;
  run;

/* Calculate the criterion mean, use that to:
    calculate the equivalence region for the intercept
    calculate adjusted criterion and surrogate values */
    
data overall2;
  merge overallmean center;
  /* should be same number of observations in the two data sets */

  critmean = criterion - center; 

  equiv = 0.10 * critmean;   
    /* Robinson's suggested equivealence bound for the intercept: 
        +/- 10% of overall criterion mean */
  call symput('equiv', equiv);
    /* and save it as a macro variable */
  
  cCriterion = center;
  cSurrogate = surrogate - critmean;
/* cCriterion is the adjusted Criterion = (criterion - overall criterion mean) */
/* cSurrogate is the adjusted Surrogate = (surrogate - overall criterion mean) */
run;

/* Easy to use the confidence interval method */
/* output includes the 90% confidence for each parameter */

proc reg data=overall2;
robinson:  model surrogate = criterion /clb alpha=0.1;
  /* Robinson's regression equivalence test */
H0slope:  test criterion = 1;  /* Traditional test of slope = 1 */

XYadj:     model cSurrogate = cCriterion / clb alpha=0.1;
  /* The adjusted test, where the intercept is the average surrogate
      at the average criterion */
title 'Multiple activity equivalence';
quit;

/* If you want the equivalence p-value, need to do a bit more processing */

/* Have to start with the intercept and slope estimates and their se's.
/* various ways to get that information */
/* this is one that doesn't require merging two data sets */

proc glm data=overall2;
  model cSurrogate = cCriterion;
  estimate 'intercept' intercept 1;
  estimate 'slope' cCriterion 1;
  ods output estimates=ests;
  run;
/* This re-fits the regression and saves the estimate and se for the intercept and slope */

/* Use a data step to compute the t statistics and p-values */
/* Code assumes that the intercept is the first line and slope is the second */

data equivtest;
  retain Pa1 Pb1;

  set ests;
  
  Nobs = 23;   /* Do need to specify the number of observations */

  if parameter = 'intercept' then do;
    Ta = (estimate - &equiv)/stderr;
  	Tb = (estimate + &equiv)/stderr;

	  Pa = 1-probt(Ta, Nobs-2);
	  Pb = probt(Tb, Nobs-2);
	  Pa1 = Pa;
	  Pb1 = Pb;  /* Retain the two p-values. Needed for overall 4 parameter test */
	  end;

  if parameter = 'slope' then do;
    Ta = (estimate - 0.9)/stderr;
	Tb = (estimate - 1.1)/stderr;

	Pa = 1-probt(Ta, Nobs-2);
	Pb = probt(Tb, Nobs-2);
	Pequiv = max(Pequiv, Pa1, Pb1, Pa, Pb);
	end;
  drop Dependent Tvalue Probt Pa1 Pb1;
  run;

proc print;
  title 'Equivalence test results';
  run;

/* Ta and Pa are the T statistic and one-sided p-value for lower one-sided test */
/* Tb and Pb are the T statistic and one-sided p-value for upper one-sided test */
/* Pequiv is the p-value for the regression equivalence test */
/*   computed as largest of the four p-values */

/* Meta regression, allowing for unequal numbers of observations per marker */

data metaregr;
  set overall2;

  weight = 1/Nsubjects;
  /* Within subject (error) variance depends on number of subjects */
  run;

proc mixed data=metaregr;
  class marker;
  model csurrogate = cCriterion /ddfm=kr;
  random marker;
  weight weight;
  estimate 'intercept' intercept 1 /cl alpha=0.1;
  estimate 'slope' cCriterion 1 /cl alpha=0.1;
  title 'Meta regression';
  run;
