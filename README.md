# InformatiCup

This repo contains our work on the InformatiCup.
The project name is SuperPerturbator.

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
mkdir /home/user/workspace/informaticup/data

# start container and bind a TCP port to host, optionally mount the data dir into the container
# use -d option (instead of -it) to run in background (deamon)
sudo docker run -it --rm \
	-p 80:80/tcp  \
	-v /home/user/workspace/informaticup/data:/informaticup/data \
	superperturbator


# to setup and run a docker container supporting nvidia cuda, execute following commands:
sudo docker built -t superperturbator-cuda -f cuda_dockerfile
sudo docker run -it --rm \
	--runtime=nvidia \
	-p 80:80/tcp  \
	-v /home/user/workspace/informaticup/data:/informaticup/data \
	superperturbator-cuda
```
## Important Notes
  - the data root directory is purely optional (it is used for caching)
  - it should however NOT be under the root of the repository directory (it will be ignored by docker)
  - the first startup will take some time since we need to fetch the training and test datasets
  - always provide fully qualified paths; it is not guaranteed to work otherwise
  - for cuda-docker, you need to be able to execute nvidia-docker. if you want to do that, please read the instructions according to your distro on [the official page](https://github.com/NVIDIA/nvidia-docker)


# Additional Information

For the caching of pip packages in the data directory to work, it must be owned by the root user (or the )