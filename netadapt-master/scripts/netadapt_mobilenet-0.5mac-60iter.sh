CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python master.py models/mobilenet/prune-by-mac-60-iter 3 224 224 \
    -im models/mobilenet/model.pth.tar -gp 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    -mi 100 -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp data/ --arch mobilenet --resume | tee NA_Mobilenet_MAC_60
