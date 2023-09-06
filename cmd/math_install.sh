cd /home/dist/math

pushd murand
bash install.sh
popd

pushd muSPARSE_dev0.1.0
bash install.sh
popd

dpkg -i muAlg_dev-0.1.1-Linux.deb

dpkg -i muThrust_dev-0.1.1-Linux.deb