# move this file to /root
apt-get update
apt-get install -y git
apt-get install -y wget
wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz
tar -xvzf "julia-1.7.3-linux-x86_64.tar.gz"
# mv julia-1.7.3/ /opt/
# ln -s /opt/julia-1.7.3/bin/julia /usr/local/bin/julia
ln -s /root/julia-1.7.3/bin/julia /usr/local/bin/julia
apt-get -y install htop
apt-get -y install tmux
