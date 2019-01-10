# InformatiCup

This repo contains our work on the InformatiCup.

# Setup

To get started, perform the following commands:

```sh
# we assume your working dir to be the following (adjust this to your path)
# /home/user/workspace

# clone repository
cd /home/user/workspace
git clone git@gitlab.com:LPirch/informaticup.git
cd informaticup

# build image
sudo docker build -t superperturbator .

# make data dir (adjust the path as you like)
# NOTE: the data dir is purely optional on the host
mkdir /home/user/workspace/informaticup/data

# start container and bind a TCP port to host, optionally mount the data dir into the container
# INFO: 	this will take some time on initial startup since we need to fetch the training
#			and test datasets; mounting this directory to the host ensures persisence
#			regardless of the container lifecycle
# INFO:		use the -it option to be able to observe the stdout of the container, use -d to
#			run in background (deamon)
# WARNING:	provide a fully qualified path here, using variables like ~ or $someVar won't
# 			work!
sudo docker run -d \
	-p 80:80/tcp  \
	-v /home/user/workspace/informaticup/data:/informaticup/data \
	superperturbator


# to setup and run a docker container supporting nvidia cuda, execute following commands:
# INFO:		you need to be able to execute nvidia-docker. if you want to do that, please
#			read the instructions on https://github.com/NVIDIA/nvidia-docker according to
#			your distro
sudo docker built -t superperturbator-cuda -f cuda_dockerfile
sudo docker run -d \
	--runtime=nvidia \
	-p 80:80/tcp  \
	-v /home/user/workspace/informaticup/data:/informaticup/data \
	superperturbator-cuda
```
