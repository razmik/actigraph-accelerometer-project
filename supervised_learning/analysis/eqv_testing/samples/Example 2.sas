/* NOTE: the data in Example 2.csv are a subset of the data used in the paper */
/*   20 individuals were randomly chosen from the full data set of 680 individuals */
/* results WILL NOT match the values reported in the paper */

proc import datafile='Example 2.csv' out=example2;
run;

/* Two one-sided tests using a ratio equivalence region given by
     values of two macro variables: equiv and Iequiv  */  
proc ttest test=ratio dist=normal tost(&equiv, &Iequiv);
  paired VO2pacer*VO2mile;
  title 'Paired ratio TOST';
  run;

