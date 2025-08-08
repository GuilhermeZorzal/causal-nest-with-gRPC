FROM python:3.9-buster AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list
RUN sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list
RUN sed -i 's|http://deb.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list
RUN apt-get -o Acquire::Check-Valid-Until=false update
RUN apt-get -qq update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get -qq install dialog apt-utils -y
RUN apt-get install apt-transport-https -y
RUN apt-get install libseccomp-dev seccomp -y
RUN apt-get install software-properties-common dirmngr -y
RUN apt-get update
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7'
# RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/debian $(env -i bash -c '. /etc/os-release; echo $VERSION_CODENAME')-cran40/" -y
RUN apt-get  update

RUN apt-get install -t buster-cran40 r-base -y --allow-unauthenticated
RUN apt-get install libssl-dev -y
RUN apt-get install libgmp3-dev  -y --allow-unauthenticated
RUN apt-get install git -y
RUN apt-get install build-essential  -y --allow-unauthenticated
RUN apt-get install libv8-dev  -y --allow-unauthenticated
RUN apt-get install libcurl4-openssl-dev -y --allow-unauthenticated
RUN apt-get install libgsl-dev -y
RUN apt-get install libxml2-dev -y --allow-unauthenticated
RUN apt-get install libharfbuzz-dev libfribidi-dev libfontconfig1-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev -y --allow-unauthenticated

RUN chmod -R 777 /usr/local/lib/R/
RUN Rscript --vanilla -e 'install.packages(c("usethis", "shiny"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages(c("Rcpp"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages(c("V8"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages(c("sfsmisc"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages(c("clue"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/randomForest/randomForest_4.6-14.tar.gz", repos=NULL, type="source", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages(c("lattice"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages(c("devtools"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages(c("MASS"), repos="http://cran.irsn.fr", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("BiocManager", repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install(c("igraph"), Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("igraph", repos="https://cloud.r-project.org", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/fastICA/fastICA_1.2-2.tar.gz", repos=NULL, type="source", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/SID/SID_1.0.tar.gz", repos=NULL, type="source", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/CAM/CAM_1.0.tar.gz", repos=NULL, type="source", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/sparsebnUtils_0.0.8.tar.gz", repos=NULL, type="source", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/ccdrAlgorithm/ccdrAlgorithm_0.0.6.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install(c("discretecdAlgorithm"), Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("discretecdAlgorithm", repos="https://cloud.r-project.org", Ncpus=4)'

# RUN Rscript --vanilla -e 'library(devtools); install_github("cran/CAM"); install_github("cran/momentchi2"); install_github("Diviyan-Kalainathan/RCIT"); install_github("cran/discretecdAlgorithm")'
RUN Rscript --vanilla -e 'install.packages("devtools", repos="https://cloud.r-project.org", Ncpus=4)'

RUN Rscript --vanilla -e 'library(devtools); install_github("cran/CAM")'
# RUN Rscript --vanilla -e 'install.packages("CAM", repos="https://cloud.r-project.org", Ncpus=4)'
RUN Rscript --vanilla -e 'library(devtools); install_github("cran/momentchi2")'
# RUN Rscript --vanilla -e 'install.packages("momentchi2", repos="https://cloud.r-project.org", Ncpus=4)'
RUN Rscript --vanilla -e 'library(devtools); install_github("Diviyan-Kalainathan/RCIT")'
# RUN Rscript --vanilla -e 'install.packages("RCIT", repos="https://cloud.r-project.org", Ncpus=4)'

RUN Rscript --vanilla -e 'library(devtools); install_github("cran/discretecdAlgorithm")'
# RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz", repos=NULL, type="source", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install(c("bnlearn", "CAM", "SID", "D2C", "pcalg", "kpcalg", "glmnet", "mboost"), Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install("bnlearn", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("bnlearn", repos="https://cloud.r-project.org", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install("CAM", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("CAM", repos="https://cloud.r-project.org", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install("SID", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("SID", repos="https://cloud.r-project.org", Ncpus=4)'
RUN Rscript --vanilla -e 'BiocManager::install("D2C", Ncpus=4)'
RUN Rscript --vanilla -e 'BiocManager::install("pcalg", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("kpcalg", repos="https://cloud.r-project.org", Ncpus=4)'
RUN Rscript --vanilla -e 'install.packages("remotes", repos="https://cloud.r-project.org")'
RUN Rscript --vanilla -e 'remotes::install_github("Diviyan-Kalainathan/RCIT")'
RUN Rscript --vanilla -e 'BiocManager::install("glmnet", Ncpus=4)'
RUN Rscript --vanilla -e 'BiocManager::install("mboost", Ncpus=4)'


# Custom below
RUN apt-get install graphviz -y
# RUN curl -sSL https://install.python-poetry.org | python3 -
RUN pip install poetry
RUN pip install notebook

RUN mkdir -p /app

COPY . /app
WORKDIR /app

# ENV PATH="${PATH}:$HOME/.local/bin"

RUN poetry env use python3.9

RUN poetry lock
RUN poetry install

WORKDIR /app

CMD ["python3"]
