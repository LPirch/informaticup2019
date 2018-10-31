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
mkdir data
wget -O train.zip "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
wget -O test.zip "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
unzip train.zip -d data
unzip test.zip -d data
rm train.zip test.zip

# fetch css files
wget -O static/css/bootstrap.min.css https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css
wget -O static/css/bootstrap.min.css https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css
wget -O static/css/bootswatch.min.css https://stackpath.bootstrapcdn.com/bootswatch/4.1.3/darkly/bootstrap.min.css

# build image
sudo docker build -t informaticup .
# start container and bind a TCP port to host
sudo docker run -d -p 80:80/tcp informaticup
```
