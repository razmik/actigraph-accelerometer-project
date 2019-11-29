/* Read in the data file: one row per individual */
proc import file='example 1.csv' out=diffs replace;
  run;

/* The difference for each individual could be computed in the SAS code */
/*   not done here because the difference is a variable in the excel file */

/* Two ways to conduct equivalence tests */

/* 1: compute the differences and specify the test statistics */

/* calculate the average and se of the differences */
/* and the 90% ci for the mean difference (confidence interval method) */
/* also need the mean for the criterion to compute the equivalence region */

proc means data=diffs n mean stderr lclm uclm alpha=0.10;
  var difference criterion;
  title 'Paired analysis of brisk walking';
  output out=diffmean mean= meanDiff meanCrit stderr=seDiff n=npair;
run;

/* Do the TOST computations to get a p-value */
data equiv;
  set diffmean;   
/* specify equivalence bounds as a fraction of "known" reference */
/* 15% = 0.15 is the subject-matter motivated definition of equivalence */
  lower = 0.15*meanCrit;
  upper = 0.15*meanCrit;

/* Two-One-Sided-Test computations */
  equivT1 = (meanDiff + lower)/seDiff;   /* T statistic for test on lower equiv boundary */
  equivT2 = (meanDiff - upper)/seDiff;   /*    and upper boundary */

  PT1 = 1-probt(equivT1, Npair-1); /* one-sided P-value reject if > lower */
  PT2 = probt(equivT2, Npair-1);   /*   and reject if < upper */
  Pequiv = max(PT1, PT2);		/* and the equivalence p-value is the larger of these two */

/* optional: do traditional t-test of no difference */
  usualT = meanDiff/seDiff;  /* T statistic for usual test: mean diff = 0 */
  PT = 2*probt(-abs(usualT), Npair-1);  /* P value for usual test */

  keep meanCrit meanDiff seDiff usualT pt equivT1 equivT2 PT1 PT2 Pequiv;
  run;

proc print data=equiv;
  format pt pt1 pt2 pequiv pvalue6.4;
  title "Equivalence test results";
  run;

/* 2: Use proc ttest, especially useful for non-paired data */
/* requires specifying absolute bounds for difference */
/* two-sample tests allow test=diff or test=ratio to choose type of region */

proc ttest data=diffs tost(-0.6512, 0.6512);
  paired criterion*surrogate;
  title "Paired t-test equivalence test";
  run;
