
# Train data generation
# # Independent features Data
load("X.Rdata")
load("y.Rdata")
# # Correlated features Data
load("X_corr.Rdata")
load("y_corr.Rdata")

# Part A
# See the model selection performance by picking the model which has the same spasity as the true model 
# Lasso
library(rqPen)
# # Independent features
betaLasso_indp <- LASSO.fit(y, X,
                           tau=.5, lambda=0.144,
                           intercept=FALSE,
                           coef.cutoff=1e-06)
cat("\n\nLasso estimator with Independent features\n\n")
print(which(!betaLasso_indp== 0))
print(betaLasso_indp[which(!betaLasso_indp == 0)])
# # Correlated features
betaLasso_corr <- LASSO.fit(y_corr, X_corr,
                           tau=.5, lambda=0.111,
                           intercept=FALSE,
                           coef.cutoff=1e-06)
cat("\n\nLasso estimator with Correlated  features\n\n")
print(which(!betaLasso_corr== 0))
print(betaLasso_corr[which(!betaLasso_corr == 0)])

# SCAD
library(ncvreg)
cat("\n\nSCAD estimator path with Indepenent  features\n\n")
betaSCAD_indp <- ncvreg(X, y, penalty="SCAD")
print(betaSCAD_indp)
cat("\n\nSCAD estimator path  with Correlated  features\n\n")
betaSCAD_corr <- ncvreg(X_corr, y_corr, penalty="SCAD")
print(betaSCAD_corr)

library(glmnet)
# Part B
# Find a optimal estimator through Cross Validation
# Lasso
# # Independent features
cv.lasso_indp=cv.glmnet(X,y)
plot(cv.lasso_indp)
betaLasso_indp <- coef(cv.lasso_indp)
betaLasso_indp_noIncept <- betaLasso_indp[2:1001,]
cat("\nCorss Validated Lasso estimator with Independent features\n\n")
print(which(!betaLasso_indp_noIncept== 0))
print(betaLasso_indp_noIncept[which(!betaLasso_indp_noIncept == 0)])
# # Correlated features
cv.lasso_corr=cv.glmnet(X_corr,y_corr)
plot(cv.lasso_corr)
betaLasso_corr <- coef(cv.lasso_corr)
betaLasso_corr_noIncept <- betaLasso_corr[2:1001,]
cat("\nCorss Validated Lasso estimator with Correlated features\n\n")
print(which(!betaLasso_corr_noIncept== 0))
print(betaLasso_corr_noIncept[which(!betaLasso_corr_noIncept == 0)])

# SCAD
# # Independent features
cvfit_indp <- cv.ncvreg(X, y, penalty="SCAD")
plot(cvfit_indp)
fit_indp <- cvfit_indp$fit
betaSCAD_indp <- fit_indp$beta[,cvfit_indp$min]
betaSCAD_indp_noIncept <- betaSCAD_indp[2:1001]
cat("\nCorss Validated SCAD estimator with Independent features\n\n")
print(which(!betaSCAD_indp_noIncept== 0))
print(betaSCAD_indp_noIncept[which(!betaSCAD_indp_noIncept == 0)])
# # Correlated features
cvfit_corr <- cv.ncvreg(X_corr, y_corr, penalty="SCAD")
plot(cvfit_corr)
fit_corr <- cvfit_corr$fit
betaSCAD_corr <- fit_corr$beta[,cvfit_corr$min]
betaSCAD_corr_noIncept <- betaSCAD_corr[2:1001]
cat("\nCorss Validated SCAD estimator with Correlated features\n\n")
print(which(!betaSCAD_corr_noIncept== 0))
print(betaSCAD_corr_noIncept[which(!betaSCAD_corr_noIncept == 0)])

# Accuracy test
# Test Data generation
n <- 100
p <- 1000
beta_0 <- matrix(0.0,p,1)
beta_0[1] = 1
beta_0[2] = -0.5
beta_0[3] = 0.7
beta_0[4] = -1.2
beta_0[5] = -0.9
beta_0[6] = 0.3
beta_0[7] = 0.55;
set.seed(1234)
library(mvtnorm)
# https://cran.r-project.org/web/packages/mvtnorm/mvtnorm.pdf
# Independent
sigma <- 1.0 * diag(1000)
X <- rmvnorm(n=100, mean=matrix(0.0,1,1000), sigma=sigma)
y <- X %*% beta_0 + 0.3 * matrix(rnorm(n),n,1)
# Correlated
sigma_corr <- 0.5 * diag(1000) + 0.5 * matrix(1.0,1000,1000) 
X_corr <- rmvnorm(n=100, mean=matrix(0.0,1,1000), sigma=sigma_corr)
y_corr <- X_corr %*% beta_0 + 0.3 * matrix(rnorm(n),n,1)


incept <- matrix(1.0, 100, 1)
inceptX <- t(rbind(t(incept),t(X)))
inceptX_corr <- t(rbind(t(incept),t(X_corr)))

# Lasso
# # Independent features
cat("\n\nLASSO Cross Validated mean square error (independent data)\n\n")
pred = inceptX %*% betaLasso_indp
print(sqrt(apply((y-pred)^2,2,mean)))
# # Correlated features
cat("\n\nLASSO Cross Validated mean square error (correlated data)\n\n")
pred = inceptX_corr %*% betaLasso_corr
print(sqrt(apply((y_corr-pred)^2,2,mean)))

# SCAD
# # Independent features
cat("\n\nSCAD Cross Validated mean square error (independet data)\n\n")
pred = inceptX %*% betaSCAD_indp
print(sqrt(apply((y-pred)^2,2,mean)))
# # Correlated features
cat("\n\nSCAD Cross Validated mean square error (correlated data)\n\n")
pred = inceptX_corr %*% betaSCAD_corr
print(sqrt(apply((y_corr-pred)^2,2,mean)))


