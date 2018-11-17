# InformatiCup

This repo contains our work on the InformatiCup.

# Setup

To get started, perform the following commands:

```sh
workspace=/path/to/your/workspace

# clone repository
cd $workspace
git clone git@gitlab.com:LPirch/informaticup.git

# setup python environment
cd informaticup
virtualenv .
source bin/activate
pip install -r pip_requirements.txt

# get data set
mkdir data && \
wget -O train.zip "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip" && \
wget -O test.zip "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip" && \
unzip train.zip -d data && \
unzip test.zip -d data && \
rm train.zip test.zip

# fetch css files
wget -O static/css/bootstrap.min.css https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css  && \
wget -O static/css/bootswatch.min.css https://stackpath.bootstrapcdn.com/bootswatch/4.1.3/darkly/bootstrap.min.css

# fetch fontawesome icons
wget -O static/css/fontawesome.zip https://use.fontawesome.com/releases/v5.5.0/fontawesome-free-5.5.0-web.zip  && \
unzip static/css/fontawesome.zip -d static/css && \
mv static/css/fontawesome-free-5.5.0-web static/css/fontawesome

# fetch JS libs
wget -O static/js/bootstrap.min.js https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js  && \
wget -O static/js/bootstrap-confirmation.min.js https://cdnjs.cloudflare.com/ajax/libs/bootstrap-confirmation/1.0.7/bootstrap-confirmation.min.js  && \
wget -O static/js/jquery.min.js https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js  && \
wget -O static/js/popper.min.js https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js


# build image
sudo docker build -t informaticup .
# start container and bind a TCP port to host
sudo docker run -d -p 80:80/tcp  -v $workspace/informaticup:/informaticup informaticup
```
