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
```
