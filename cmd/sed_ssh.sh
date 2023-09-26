# sed -i 's/^service ssh restart/#service ssh restart/' ~/.bashrc
# ulimit -n
# sed -i '107s/export PATH=$PATH:\/data\/yehua\/openmpi\/bin/export PATH=$PATH:\/home\/dist\/opt\/openmpi\/bin/' ~/.bashrc
# sed -i '108s/export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\/data\/yehua\/openmpi\/lib/export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\/home\/dist\/opt\/openmpi\/lib/' ~/.bashrc
# echo 'ulimit -l unlimited' >> ~/.bashrc
sed -i '54a root            hard    memlock         unlimited' /etc/security/limits.conf
sed -i '55a root            soft    memlock         unlimited' /etc/security/limits.conf