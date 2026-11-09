
library(INLA)
library(withr)
library(sf)
library(ggplot2)

est.GP.INLA <- function(data, spde, locations, plot=FALSE, n.samples=500, seed=0L){
  # It stimates a continuous GP using INLA
  # arg: 
  #   data= a vector with lenght=num of locations
  #   spde= SPDE object from INLA with informations such as mesh and prior parameters
  #   locations= a matrix with 2 columns where data are observed
  # values: a list 
  #   $Coeff:  coefficients of the basis expression
  #   $H:  4x1 matrix containing the parameters estimated, mu, Stdev for the Gaussian observations (nugget)  
  #        range, and Stdev(field): sigma= 1/ (4*pi*kappa^2 *tau^2) 
  #   $X.hat: fitted values 
  #   $Sd.fields: posterior sd of (intercept + field) at each mesh node
  #   $Sd.mean:   posterior sd of the spatial mean of (intercept + field)
  #   $all_est: all outputs from INLA
  # arg n.samples: joint posterior draws used for Sd.fields; 0 disables it
  # arg seed: 0 (default) runs INLA as normal, multi-threaded with random
  #   posterior sampling. Any other value runs INLA single-threaded in
  #   classic mode and seeds the sampling, which makes repeated runs
  #   bit-for-bit identical. Single-threading alone is not enough.
  
  reproducible <- seed != 0
  # Multi-threaded INLA is not deterministic between runs
  threads <- if (reproducible) "1:1" else inla.getOption("num.threads")
  if (reproducible) {
    # Classic mode only takes effect as a global option, not as an argument
    # to inla(), so set it here and put it back when the function exits
    old.mode <- inla.getOption("inla.mode")
    inla.setOption(inla.mode = "classic")
    on.exit(inla.setOption(inla.mode = old.mode), add = TRUE)
  }

  mesh <- spde$mesh
  A <- inla.spde.make.A(mesh= mesh,
                        loc=locations)
  
  mesh.index = inla.spde.make.index(name="field",
                                    n.spde=mesh$n)
  
  # Param=saves the mean parameters estimated: mu0_t, 
  # Stdev for the Gaussian observations (nugget), range, Stdev(field), 
  # and log(tau), log(kappa) 
  Param <- matrix(NA,4,1) 
  # Coeff.fields= saves coefficients of \phi_k
  Coeff.fields <- matrix(NA,mesh$n, 1)
  
  # INLA model ---------------------------------------
  stack.est <- inla.stack(data=list(y=c(data)), 
                            A=list(A),
                            effects=list(c(mesh.index,list(intercept=1))),
                            tag="est")
  ## Formula 
  formula <- y ~ -1 + intercept + f(field, model=spde)
  # 
  est_output <- inla(formula,  family="gaussian", 
                       data=inla.stack.data(stack.est,spde=spde), 
                       control.predictor=list(A=inla.stack.A(stack.est),compute=TRUE,link=1),
                       control.compute=list(cpo=TRUE, config=TRUE, dic=TRUE),
                       verbose=FALSE,
                       control.mode=list(theta=c(1.3,5,-.8), restart=TRUE),
                       control.inla=list(strategy='gaussian',int.strategy='eb'),
                       num.threads=threads)
  
  # Extract parameters estimated ----------------------------
  # mu0_t
  Param[1,1] <- round(est_output$summary.fixed$mean,4)
  #inla.emarginal(function(x) x, est_output$marginals.fixed[[1]])
  # Stdev for the Gaussian observations (nugget)  
  Param[2,1] <- 1/round(est_output$summary.hyperpar$mean[1],4)
  # inla.emarginal(function(x) 1/x, est_output$marginals.hyper[[1]])
  #range
  Param[3,1] <- round(est_output$summary.hyperpar$mean[2],4)
  #inla.emarginal(function(x) x, output.result$marginals.range.nominal[[1]]) # rho= sqrt(8)/kappa
  
  #Stdev(field): sigma= 1/ (4*pi*kappa^2 *tau^2)
  Param[4,1] <- round(est_output$summary.hyperpar$mean[3],4)
  #sqrt(inla.emarginal(function(x) x, output.result$marginals.variance.nominal[[1]]))
    
  Coeff.fields[ ,1] <- round(est_output$summary.random$field$mean,4)

  rownames(Param) <- c('mu','StdevG','range','StdevF')
  
  #values fitted 
  X.hat <- Param[1,1] + as.vector(A %*% Coeff.fields[ ,1]) 

  # Uncertainty of the fitted field ------------------------------------
  # A fitted node is intercept + field, and those two are strongly (and
  # negatively) correlated: the field absorbs the overall level, so the pair
  # trade off against one another. Their marginal standard deviations cannot
  # therefore be combined by any rule - not by summing, and not in quadrature
  # either. The joint posterior is needed, and control.compute(config=TRUE)
  # above already makes it available.
  Sd.fields <- NULL
  Sd.mean <- NULL
  if (n.samples > 0) {
    if (reproducible) set.seed(seed)
    samples   <- inla.posterior.sample(n = n.samples, est_output,
                                       seed = if (reproducible) seed else 0L)
    row_names <- rownames(samples[[1]]$latent)
    idx_field <- grep("^field", row_names)
    idx_int   <- grep("^intercept", row_names)
    if (length(idx_field) == mesh$n && length(idx_int) == 1) {
      # draws: mesh$n rows by n.samples columns
      draws     <- sapply(samples,
                          function(s) s$latent[idx_field] + s$latent[idx_int])
      Sd.fields <- apply(draws, 1, sd)
      # sd of the catchment mean, taken across draws so the spatial
      # correlation between nodes is carried rather than assumed away
      Sd.mean   <- sd(colMeans(draws))
    } else {
      warning(paste("est.GP.INLA: could not locate field/intercept rows in the",
                    "posterior sample; Sd.fields not computed"))
    }
  }
  # plot
  pl <- ggplot(data.frame(lon = locations[,1],
                          lat = locations[,2],
                          hatx = X.hat)) +
    geom_tile(aes(x=lon, y=lat, fill=hatx), width=50, height=50) +
    coord_equal() +
    xlab("Longitude") +
    ylab("Latitude") +
    scale_fill_viridis_c(begin = 0, end = 1,"hat x")
  
  if(plot){
    print(pl)
  }
  
  return(list(Coeff=Coeff.fields, H=Param, X.hat=X.hat,
              Sd.fields=Sd.fields, Sd.mean=Sd.mean,
              all.est=est_output))
}

