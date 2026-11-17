* ============================================================================.
* Analysis Syntax - SPSS 26.0.
* Analytic sample: N = 205 valid cases; composite scores standardized;
* no demographic covariates (Section 3.4).
* process.sps (Hayes PROCESS v4.1) must be installed before section 6-7.
* ============================================================================.

* ---------- 0. Standardized composites (all subsequent analyses use these) ----.
DESCRIPTIVES VARIABLES=ps_mean pcj_mean ts_mean si_mean se_mean
  /SAVE.
* Creates Zps_mean Zpcj_mean Zts_mean Zsi_mean Zse_mean.

* ---------- 1. Demographics (Table 1) ----------.
FREQUENCIES VARIABLES=gender grade major hukou onlychild cadre class.

* ---------- 2. Descriptives (Section 4.1, Figure 2) ----------.
DESCRIPTIVES VARIABLES=ps_mean pcj_mean ts_mean si_mean se_mean
  /STATISTICS=MEAN STDDEV MIN MAX SKEWNESS KURTOSIS.

* ---------- 3. Reliability: Cronbach's alpha (Section 4.1) ----------.
* McDonald's omega was computed in R with psych::omega() (see README).
RELIABILITY /VARIABLES=ps1 TO ps16  /SCALE('PS')  ALL /MODEL=ALPHA.
RELIABILITY /VARIABLES=pcj1 TO pcj20 /SCALE('PCJ') ALL /MODEL=ALPHA.
RELIABILITY /VARIABLES=ts1 TO ts17  /SCALE('TS')  ALL /MODEL=ALPHA.
RELIABILITY /VARIABLES=si1 TO si15  /SCALE('SI')  ALL /MODEL=ALPHA.
RELIABILITY /VARIABLES=se1 TO se18  /SCALE('SE')  ALL /MODEL=ALPHA.

* ---------- 4. Correlations (Table 2) ----------.
CORRELATIONS
  /VARIABLES=Zps_mean Zpcj_mean Zts_mean Zsi_mean Zse_mean
  /PRINT=TWOTAIL
  /STATISTICS DESCRIPTIVES.

* ---------- 5. Demographic difference tests (Section 4.2, Figures 4-6) ----.
T-TEST GROUPS=gender(1 2)    /VARIABLES=ps_mean pcj_mean ts_mean si_mean se_mean.
T-TEST GROUPS=hukou(1 2)     /VARIABLES=ps_mean pcj_mean ts_mean si_mean se_mean.
T-TEST GROUPS=onlychild(1 2) /VARIABLES=ps_mean pcj_mean ts_mean si_mean se_mean.
T-TEST GROUPS=cadre(1 2)    /VARIABLES=ps_mean pcj_mean ts_mean si_mean se_mean.
ONEWAY ps_mean pcj_mean ts_mean si_mean se_mean BY grade
  /STATISTICS DESCRIPTIVES HOMOGENEITY.
ONEWAY ps_mean pcj_mean ts_mean si_mean se_mean BY major
  /STATISTICS DESCRIPTIVES HOMOGENEITY.

* ---------- 6. ICC(1) for classroom nesting (end of Section 3.4) ----------.
ONEWAY Zps_mean Zpcj_mean Zts_mean Zsi_mean Zse_mean BY class
  /STATISTICS DESCRIPTIVES.
* ICC(1) = (MSB - MSW) / (MSB + (k-1)*MSW); k = mean class size = 205/8 = 25.6.

* ---------- 7. Simple mediation - Path 1: PS -> TS -> SE (Tables 4-5) -------.
INSERT FILE='C:\\PROCESS_v4.1\\process.sps'.
PROCESS vars = Zse_mean Zps_mean Zts_mean
  /y = Zse_mean
  /x = Zps_mean
  /m = Zts_mean
  /model = 4
  /boot = 5000
  /bc = 1
  /conf = 95
  /total = 1.

* ---------- 8. Simple mediation - Path 2: PCJ -> SI -> SE (Tables 4-5) -----.
PROCESS vars = Zse_mean Zpcj_mean Zsi_mean
  /y = Zse_mean
  /x = Zpcj_mean
  /m = Zsi_mean
  /model = 4
  /boot = 5000
  /bc = 1
  /conf = 95
  /total = 1.

* ---------- 9. Integrated regressions (Table 6) ----------.
REGRESSION /MISSING LISTWISE /STATISTICS COEFF OUTS R ANOVA
  /DEPENDENT=Zts_mean /METHOD=ENTER Zps_mean Zpcj_mean.
REGRESSION /MISSING LISTWISE /STATISTICS COEFF OUTS R ANOVA
  /DEPENDENT=Zsi_mean /METHOD=ENTER Zps_mean Zpcj_mean.
REGRESSION /MISSING LISTWISE /STATISTICS COEFF OUTS R ANOVA
  /DEPENDENT=Zse_mean /METHOD=ENTER Zps_mean Zpcj_mean Zts_mean Zsi_mean.

* ---------- 10. Parallel mediation, two configurations (Table 7) -------------.
* Config A: focal X = PS, other predictor PCJ entered as covariate.
PROCESS vars = Zse_mean Zps_mean Zts_mean Zsi_mean Zpcj_mean
  /y = Zse_mean
  /x = Zps_mean
  /m = Zts_mean Zsi_mean
  /c = Zpcj_mean
  /model = 4
  /boot = 5000
  /bc = 1
  /conf = 95
  /contrast = 1.
* Config B: focal X = PCJ, other predictor PS entered as covariate.
PROCESS vars = Zse_mean Zpcj_mean Zts_mean Zsi_mean Zps_mean
  /y = Zse_mean
  /x = Zpcj_mean
  /m = Zts_mean Zsi_mean
  /c = Zps_mean
  /model = 4
  /boot = 5000
  /bc = 1
  /conf = 95
  /contrast = 1.
* Specific indirect effects:
*   PS->TS->SE and PS->SI->SE from Config A output;
*   PCJ->TS->SE and PCJ->SI->SE from Config B output;
*   "contrast = 1" gives the PS->TS->SE vs PS->SI->SE (and PCJ->SI->SE vs
*   PCJ->TS->SE) contrasts; the cross-pair contrast (PS->TS->SE vs PCJ->SI->SE)
*   is computed by subtracting the corresponding bootstrap estimates.

* ---------- 11. Harman single-factor test (Section 4.1) ------------------.
FACTOR
  /VARIABLES ps1 TO ps16 pcj1 TO pcj20 ts1 TO ts17 si1 TO si15 se1 TO se18
  /MISSING LISTWISE
  /ANALYSIS ps1 TO ps16 pcj1 TO pcj20 ts1 TO ts17 si1 TO si15 se1 TO se18
  /PRINT INITIAL EXTRACTION
  /CRITERIA FACTORS(1) ITERATE(25)
  /EXTRACTION PCF
  /ROTATION NOROTATE
  /METHOD=CORRELATION.
* % of variance of the first unrotated factor = 28.6% (threshold 40%).
