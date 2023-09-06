# pip uninstall torch_musa -y

cd /home/torch_musa
mv /home/torch_musa/torch_patches/distributed_c10d.py.patch /home
mv /home/torch_musa/torch_musa/csrc/distributed/ProcessGroupMCCL.cpp /home

cp /home/dist/patch/* /home/torch_musa/torch_patches/
cp /home/dist/llm/PGMCCL.cpp /home/torch_musa/torch_musa/csrc/distributed/ProcessGroupMCCL.cpp

cd /home/torch_musa
# rm -rf build
bash build.sh