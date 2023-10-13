# CSGD_error_model
CSGD error model for Satellite precipitation uncertainty quantification. 

The censored shifted gamma distribution (CSGD) based satellite precipitation error modeling framework was used to model the pixel-scale uncertainty of Satellite precipitation. CSGD modifies the conventional two-parameter gamma distribution with shifting and left-censoring. By doing so the error model can depict both precipitation occurrence and magnitude. This framework is flexible, capturing the probability of precipitation, central tendency (i.e. median, mean), and uncertainty using the three parameters of CSGD (Scheuerer et al., 2015): the mean, standard deviation, and shift. The shift allows the model to describe the probability of both zero and positive precipitation, with the cumulative distribution function evaluated at zero being equal to the probability of zero precipitation. The CSGD error model generates conditional distributions of rainfall via a nonlinear regression, whereby three CSGD parameters can be conditioned on satellite precipitation intensity (i.e., IMERG Early) and other time-varying covariates, such as Wetted Area Ratio (WAR; the percentage of pixels with positive precipitation in each box centered on each pixel). 

Input: satellite precipitation observations, ground truth, and other covariates.

Output: CSGD parameters and related probability distributions

More details about the algorithm can be seen in 

Wright, D. B., Kirschbaum, D. B., & Yatheendradas, S. (2017). Satellite Precipitation Characterization, Error Modeling, 
and Error Correction Using Censored Shifted Gamma Distributions. 
Journal of Hydrometeorology, 18(10), 2801-2815. https://doi.org/10.1175/JHM-D-17-0060.1

Scheuerer, M., & Hamill, T. M. (2015). Statistical Postprocessing of Ensemble Precipitation Forecasts
by Fitting Censored, Shifted Gamma Distributions. 
Monthly Weather Review, 143(11), 4578-4596. https://doi.org/10.1175/MWR-D-15-0061.1

For any questions, feel free to contact Kaidi Peng (kaidi.peng@wisc.edu)

