FROM python:3.9-buster as base


# COMMENTING EVERITHING TO MAKE CONTAINER LIGHTER. THE IMPORTANT NOW IS TO TEST IF THE GRPC WORKS
# FROM HERE =====================================================================================

# ENV DEBIAN_FRONTEND noninteractive
# RUN apt-get -qq update
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
# RUN apt-get -qq install dialog apt-utils -y
# RUN apt-get install apt-transport-https -y
# RUN apt-get install libseccomp-dev seccomp -y
# RUN apt-get install software-properties-common dirmngr -y
# RUN apt-get update
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7'
# # RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/debian $(env -i bash -c '. /etc/os-release; echo $VERSION_CODENAME')-cran40/" -y
# RUN apt-get  update
#
# RUN apt-get install -t buster-cran40 r-base -y --allow-unauthenticated
# RUN apt-get install libssl-dev -y
# RUN apt-get install libgmp3-dev  -y --allow-unauthenticated
# RUN apt-get install git -y
# RUN apt-get install build-essential  -y --allow-unauthenticated
# RUN apt-get install libv8-dev  -y --allow-unauthenticated
# RUN apt-get install libcurl4-openssl-dev -y --allow-unauthenticated
# RUN apt-get install libgsl-dev -y
# RUN apt-get install libxml2-dev -y --allow-unauthenticated
# RUN apt-get install libharfbuzz-dev libfribidi-dev libfontconfig1-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev -y --allow-unauthenticated
#
# RUN chmod -R 777 /usr/local/lib/R/
# RUN Rscript --vanilla -e 'install.packages(c("usethis", "shiny"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages(c("Rcpp"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages(c("V8"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages(c("sfsmisc"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages(c("clue"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/randomForest/randomForest_4.6-14.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages(c("lattice"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages(c("devtools"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages(c("MASS"), repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("BiocManager", repos="http://cran.irsn.fr", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install(c("igraph"), Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/fastICA/fastICA_1.2-2.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install(c("bnlearn", "CAM", "SID", "D2C", "pcalg", "kpcalg", "glmnet", "mboost"), Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/SID/SID_1.0.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/CAM/CAM_1.0.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/sparsebnUtils_0.0.8.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/ccdrAlgorithm/ccdrAlgorithm_0.0.6.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install(c("discretecdAlgorithm"), Ncpus=4)'
#
# RUN Rscript --vanilla -e 'library(devtools); install_github("cran/CAM"); install_github("cran/momentchi2"); install_github("Diviyan-Kalainathan/RCIT"); install_github("cran/discretecdAlgorithm")'
# # RUN Rscript --vanilla -e 'install.packages("https://cran.irsn.fr/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz", repos=NULL, type="source", Ncpus=4)'
# RUN Rscript --vanilla -e 'BiocManager::install(c("bnlearn", "CAM", "SID", "D2C", "pcalg", "kpcalg", "glmnet", "mboost"), Ncpus=4)'
#
# # Custom below
# RUN apt-get install graphviz -y
# # RUN curl -sSL https://install.python-poetry.org | python3 -
# RUN pip install poetry
#
# TO HERE =====================================================================================
# Installing grpc dependencies
#
RUN pip install grpcio
RUN pip install grpcio-tools

RUN mkdir -p /app

COPY . /app
WORKDIR /app

# ENV PATH="${PATH}:$HOME/.local/bin"

# HERE ALSO ====================================================================================
# RUN poetry env use python3.9
# RUN poetry install

WORKDIR /app

CMD ["python3", "src/server.py"]
