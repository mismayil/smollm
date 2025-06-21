#!/bin/bash

NFS_HOME="/scratch/users/ismayilz"
HOME="/home/ismayilz"

# symlink folders
rm -rf ~/.ssh
ln -s $NFS_HOME/.ssh $HOME/.ssh
rm -rf ~/.cache
ln -s $NFS_HOME/.cache $HOME/.cache

# export NFS_HOME
echo "export NFS_HOME=$NFS_HOME" >> ~/.bashrc
# echo "export PROJECT_HOME=$NFS_HOME/project-oracl" >> ~/.bashrc

source ~/.bashrc
exec /bin/bash