# pip uninstall torch_musa -y

cd /home/torch_musa
# mv /home/torch_musa/torch_patches/distributed_c10d.py.patch /home
# mv /home/torch_musa/torch_musa/csrc/distributed/ProcessGroupMCCL.cpp /home

cp /home/dist/patch/* /home/torch_musa/torch_patches/
# cp /home/dist/llm/PGMCCL.cpp /home/torch_musa/torch_musa/csrc/distributed/ProcessGroupMCCL.cpp

cd /home/torch_musa/torch_musa/csrc/core

sed -i '459s/TORCH_INTERNAL_ASSERT(params.err == musaErrorMemoryAllocation);/\/\/TORCH_INTERNAL_ASSERT(params.err == musaErrorMemoryAllocation);/' Allocator.cpp
# sed -i '1142s/TORCH_CHECK(p.err == musaSuccess, "Musa Tensor Allocate failed!");/\/\/TORCH_CHECK(p.err == musaSuccess, "Musa Tensor Allocate failed!");/' Allocator.cpp
sed -i '1142s/TORCH_CHECK(p.err == musaSuccess, "Musa Tensor Allocate failed!");/\/\/TORCH_CHECK(p.err == musaSuccess, "Musa Tensor Allocate failed!");/' Allocator.cpp

cd /home/torch_musa/torch_musa/csrc/distributed
# git restore ProcessGroupMCCL.cpp
# sed -i '1114s/streamVal.push_back(c10::musa::getDefaultMUSAStream());/streamVal.push_back(c10::musa::getStreamFromPool(options_->is_high_priority_stream));/' ProcessGroupMCCL.cpp

cd /home/torch_musa

# rm -rf build
sed -i '19s/MUSA_ARCH=21/MUSA_ARCH=22/' build.sh


bash build.sh -m