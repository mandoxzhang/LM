# cd /usr/local/musa/include
# sed -i "66s/\/\/#define __MUSA_USE_NEW_FP16__/#define __MUSA_USE_NEW_FP16__/" musa_fp16.h

# cd /home/dist/dev1.5.0/musa_toolkits_install
# bash install.sh

# cd /home/dist/muDNN/build/mp_22/mudnn
# bash install_mudnn.sh

cd /home/dist/perf/mccl_2.11.4-1+musa1.0_x86_64
bash install.sh

echo 'done'