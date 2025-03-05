# ============================
# 📌 Step 1: Set Up CRAN Repository & Library Path
# ============================
r = getOption("repos") 
r["CRAN"] = "https://cloud.r-project.org"
options(repos = r)

# Create local user library path (not present by default)
dir.create(path = Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)

# ============================
# 📌 Step 2: Fix Matrix Version Issue (Ensures Compatibility)
# ============================
tryCatch({
    install.packages("Matrix")
}, error = function(e) {
    message("Matrix failed. Installing an older compatible version...")
    install.packages("https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.5-3.tar.gz",
                     repos = NULL, type = "source")
})

# ============================
# 📌 Step 3: Install Packrat & Initialize Environment
# ============================
install.packages("packrat")
packrat::init(infer.dependencies=FALSE)
packrat::set_opts(local.repos = c("."))

# ============================
# 📌 Step 4: Install Local Packages (e.g., dummies)
# ============================
packrat::install_local("dummies_1.5.6.tar.gz")  # dummies is removed from CRAN

# ============================
# 📌 Step 5: Install Essential Dependencies First
# ============================
install.packages(c(
    "cli", "colorspace", "ellipsis", "magrittr", "pillar", "gtable",
    "fansi", "utf8", "rlang", "pkgconfig", "tidyselect", "stringi", 
    "tzdb", "hms", "glue", "isoband", "mgcv", "vctrs", "withr", "scales", 
    "lifecycle", "munsell", "ggplot2", "dplyr", "tidyr", "readr", "purrr",
    "tibble", "forcats", "DiceKriging", "MASS", "Rcpp", "RcppEigen", "nloptr", 
    "R6", "progress", "RColorBrewer", "bit64", "bit", "clipr", "crayon",
    "digest", "farver", "generics", "labeling", "prettyunits", "vroom",
    "viridisLite", "RobustGaSP", "DiceEval"
), dependencies = TRUE)

# ============================
# 📌 Step 6: Fix Possible CRAN Version Issues
# ============================
# If these packages fail to install, try installing an older version
tryCatch({
    install.packages("cpp11")
}, error = function(e) {
    message("cpp11 failed. Installing older version from archive...")
    install.packages("https://cran.r-project.org/src/contrib/Archive/cpp11/cpp11_0.2.6.tar.gz", 
                     repos = NULL, type = "source")
})

tryCatch({
    install.packages("tzdb")
}, error = function(e) {
    message("tzdb failed. Installing older version from archive...")
    install.packages("https://cran.r-project.org/src/contrib/Archive/tzdb/tzdb_0.1.0.tar.gz", 
                     repos = NULL, type = "source")
})

tryCatch({
    install.packages("purrr")
}, error = function(e) {
    message("purrr failed. Installing older version from archive...")
    install.packages("https://cran.r-project.org/src/contrib/Archive/purrr/purrr_0.3.4.tar.gz", 
                     repos = NULL, type = "source")
})

# ============================
# 📌 Step 7: Install Emulandice
# ============================
packrat::install("emulandice")

# ============================
# 📌 Step 8: Save Packrat Snapshot
# ============================
packrat::snapshot()
