CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python master.py models/mobilenet/prune-by-latency-60 3 224 224 \
    -im models/mobilenet/model.pth.tar -gp 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    -mi 60 -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 -lt latency_lut/lut_mobilenet.pkl \
    -dp data/ --arch mobilenet > NA_Mobilenet_Latency_60
