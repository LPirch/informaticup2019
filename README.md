# SuperPerturbator
> You don't have to wait for later  
> To fool a classificator  
> Ask your local hacker trader  
> For the SuperPerturbator <sup>[1](#manitu)</sup>


This repo contains our work on the InformatiCup, solving the challenge of adversarial example creation in a black-box scenario.

# Setup

The software is distributed as a git repository and a ready-to-use docker container.
To get started, perform the commands corresponding to your use case.

## Fetching the Sources
To be able to fetch from our repository, you need to provide us your public SSH key to grant you access.
```sh
# clone the git repo
git clone git@gitlab.com:LPirch/informaticup.git
cd informaticup
```

Alternatively, you can import the repo from a tar ball:
```sh
# import from tar
cd /home/user/location/of/targz-file
tar -xzf superperturbator-repo.tar.gz
```

## Building from Source
```sh
# change to the local repository directory
cd /home/user/workspace/informaticup
# build the docker image
sudo docker build -t superperturbator .
```

## Import a pre-built Docker Image
To import a docker image which has been distributed as tar archive, execute the following command:
```sh
sudo docker load --input superperturbator-image.tar 
```

## Prepare a Data Directory (optional)
We recommend saving the dataset and internal program state in a separate directory on the host.
The advantage of this is that restarting the container won't require download all the data again and also, the program state is not lost on container termination.  
This decouples the program state from the container lifecycle.

```sh
# make data dir (adjust the path as you like)
mkdir /home/user/data
```

Please make sure that the data directory is not located in the directory of the repository if it has been used to create the docker image.
Due to internal restrictions on docker contexts, this directory will be ignored otherwise.

## Run the Container
To start the container, you must have built or imported the docker image.
Check the output of 
```sh 
sudo docker images
```
whether it contains the superperturbator image.

Next, you can start it by typing
```sh
sudo docker run -it --rm \
	-p 80:80/tcp  \
	-v /home/user/workspace/informaticup/data:/informaticup/data \
	superperturbator
```
The -p flag binds a TCP port to the host such that it can be reached in the user's web browser.  
The -v flag is optional and mounts the data directory into the container.  
Optionally, you can replace the -it by -d which runs the docker container as a daemon.

## Run a nvidia cuda Container
Our software supports the usage of nvidia cude which greatly speeds up computational tasks like training or attacking a model.  
However, this technique is not officially supported since it is heavily dependent on the machine executing it.  
If you want to use this feature nevertheless, use the following commands to use a cuda docker image.
```sh
sudo docker build -t superperturbator-cuda -f cuda_dockerfile
sudo docker run -it --rm \
	--runtime=nvidia \
	-p 80:80/tcp  \
	-v /home/user/workspace/informaticup/data:/informaticup/data \
	superperturbator-cuda
```
## Important Notes
  - the first startup will take some time since we need to fetch the training and test datasets
  - always provide fully qualified paths; it is not guaranteed to work otherwise
  - for cuda-docker, you need to be able to execute nvidia-docker. if you want to do that, please read the instructions according to your distro on [the official page](https://github.com/NVIDIA/nvidia-docker)
  - cuda-docker is not officially supported; use it at own risk
  - the files and directories created in the data directory will be owned by the user executing the docker daemon (usually root); do not modify the file permissions because docker won't be able to use them otherwise


<hr>
<a name="manitu">1</a>: Adapted from the superperforator <a href=https://www.youtube.com/watch?v=fuZN5mVNnbc>advertisement</a> in the movie "The shoe of Manitou" ("Der Schuh des Manitu")