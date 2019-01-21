
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, ggplot2)

d <- 1 # Dimensions used for reconstruction


# PCA ---------------------------------------------------------------------

pacman::p_load(FactoMineR)

data(decathlon)

# Only focus on three of the variables for simplicity
X <- as.matrix(decathlon[, c("Long.jump", "Shot.put", "High.jump")])

# Depending on the implementation of PCA, packages use centred or scaled 
# versions of the datasets
X_cent <- sweep(X, MARGIN = 2, colMeans(X), "-")
X_scal <- sweep(X_cent, MARGIN = 2, apply(X_cent, MARGIN = 2, sd) * 40 / 41, "/")

ggplot(as.data.frame(X_scal), aes(Long.jump, Shot.put)) + 
  geom_point()

# Have a look what the base R package prcomp does
prcomp(X, center = FALSE)     # Uncentered, unscaled
prcomp(X)                     # Centered, unscaled
prcomp(X_cent)                # Centered, unscaled
prcomp(X_cent, scale. = TRUE) # Centered, scaled
prc_s <- prcomp(X_scal)       # Centered, scaled


# Try to get the same with eigen() and svd()
(e <- eigen(t(X) %*% X))    # Eigenvalus and -vectors of the (uncentered) covariance matrix
svd(X)                      # Singular vectors are the square roots of the eigenvalues

(ec <- eigen(t(X_cent) %*% X_cent))
svd(X_cent)

(es <- eigen(t(X_scal) %*% X_scal))
svd(X_scal)


# Now see what FactoMineR gives us
res.pca_c <- PCA(X, scale.unit = FALSE)
near(abs(res.pca_c$ind$coord), abs(X_cent %*% ec$vectors))

res.pca_s <- PCA(X)
near(abs(res.pca_s$ind$coord), abs(X_scal %*% es$vectors)) 

# NOTE: They seem to do something slightly different with scaling, 
#       see escart.type in the result and ec.tab() defined in PCA()



# Reconstruct the original
near(prc_s$x %*% t(prc_s$rotation), X_scal)
prc_s$x[, 1:d] %*% t(prc_s$rotation[, 1:d])


PCA_raw <- res.pca_s$ind$coord[, 1:d] %*% t(sweep(res.pca_s$var$coord, 2, sqrt(res.pca_s$eig[, 1]), "/")[, 1:d]) 
PCA_recon <- sweep(sweep(PCA_raw, 2, res.pca_s$call$ecart.type, "*"), 2, res.pca_s$call$centre, "+") # Need to scale and add centre
all(near(PCA_recon, reconst(res.pca_s, ncp = d))) # Same results as method provided by FactoMineR





# CA ----------------------------------------------------------------------

pacman::p_load(FactoMineR)

data(tea)

# First run CA on a 2 variable table, formulas were partially taken from 
# https://en.wikipedia.org/wiki/Correspondence_analysis, but there seem to 
# be errors.
X <- tea[, c("Tea", "How")]

C <- table(X) %>% as.data.frame.matrix() %>% as.matrix()

n_C <- sum(C)
w_m <- as.vector(C %*% rep(1L, ncol(C)) / n_C) # scaled row margins, called row weights in Wikipedia
w_n <- as.vector(t(rep(1L, nrow(C))) %*% C / n_C) # scaled col margins (aka column weights)

S <- C / n_C   # table of relative porportions

M <- (S - w_m %*% t(w_n)) / w_m %*% t(w_n)  # Chi-square components
# Note: M differs from the description on wikipedia through the 
#       extra division by w_m %*% t(w_n). This is in line with 
#       the implimentation in FactoMineR and, for each cell, 
#       represents (O - E) / E


# Get the results via Generalised Singular Value Decomposition
res.ca <- svd.triplet(M, w_m, w_n)

# NOTE: alternative functions for GSVD exist, e.g. MFAg::GSVD


# Compare to FactoMineR
res.ca_FMR <- CA(table(X))

res.ca$vs ^ 2 # Eigenvalues of the covariance matrix of the table under constraints
res.ca_FMR$eig

sweep(res.ca$U, 2, res.ca$vs, "*")   # Coordinates of row categories
res.ca_FMR$row$coord

sweep(res.ca$V, 2, res.ca$vs, "*")  # Coordinates of column categories
res.ca_FMR$col$coord


# Reconstruct the original table

# In the singular value decomposition M = USigmaV*
U <- cbind(res.ca$U, 0) # Dimensions seem to be dropped because one eigenvector is 0
V <- cbind(res.ca$V, 0) # Same here
E <- diag(c(res.ca$vs, 0)) # Plus the 0 eigenvector

near(M, U %*% E %*% t(V))

U_re <- sweep(res.ca_FMR$row$coord, 2, sqrt(res.ca_FMR$eig[, 1]), "/")
U_re <- cbind(U_re, 0)

V_re <- sweep(res.ca_FMR$col$coord, 2, sqrt(res.ca_FMR$eig[, 1]), "/")
V_re <- cbind(V_re, 0)

E_re <- diag(c(res.ca$vs, 0))

M_re.full <- U_re %*% E_re %*% t(V_re)

d <- 1
M_re.part <- U_re[, 1:d, drop = FALSE] %*% E_re[1:d, 1:d] %*% t(V_re[, 1:d])


(M_re.full * w_m %*% t(w_n) + w_m %*% t(w_n)) * n_C

(M_re.part * w_m %*% t(w_n) + w_m %*% t(w_n)) * n_C
reconst(res.ca_FMR, d)



# Note: Wikipedia states a few weird divergences from the above 
#       approach, particulalry the measures defined below

M <- (S - w_m %*% t(w_n))

W_m <- diag(1 / as.vector(w_m))
W_n <- diag(1 / as.vector(w_n))

F_m <- W_m %*% U %*% E
F_n <- W_n %*% V %*% E




# MCA ---------------------------------------------------------------------

pacman::p_load(caret)

X <- tea[, 1:3]

res.mca <- MCA(X)
reconst(res.mca, d) # ERROR: no method exists to reconstruct MCA

# However, MCA is just CA on the indicator table:
# http://www.statsoft.com/textbook/correspondence-analysis
X_ind <- predict(caret::dummyVars(~ breakfast + tea.time + evening, X), X)
res.ca <- CA(X_ind)

res.mca$eig
res.ca$eig[1:nrow(res.mca$eig),] # NOTE: has more rows but those do not 
                                 # explain anything due to the linear contraints

# This allows us now to reconstruct the original indicator table
# For exact calculation see section on CA above
reconst(res.ca, d)




