pip uninstall -y colossalai

# cd /home/dist/colossalai_musa

cd /usr/local/musa/include
sed -i "66s/#define __MUSA_USE_NEW_FP16__/\/\/#define __MUSA_USE_NEW_FP16__/" musa_fp16.h
# sed -i "66s/\/\/#define __MUSA_USE_NEW_FP16__/#define __MUSA_USE_NEW_FP16__/" musa_fp16.h

cd /home/dist/cai/ColossalAI
MUSA_CPU=1 pip install -e .