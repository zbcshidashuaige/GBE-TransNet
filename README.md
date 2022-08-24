# GBE-TransNet
This paper is being submitted for Remote Sensing named "GBE-TransNet: A Muti-branch Network with Gated Boundary Enhancing Learning and Transformer for Semantic Segmentation of Remote Sensing Images". Any questions could be answered if we can. The E-mail is 1091007069@qq.com.

For the network, it has two branch (regular branch and boundary branch). In regular branch, FCN and Transformer are combined. In boundary branch, the lost boundary information should be recovered.


In this work, you should have a GPU whose video memory is over 8 GB. The input images and labels should be 512×512 (pixels * pixels). If not, you should change the relative parameters in Vit and GBE. (best: 3090 or 2080Ti)

For w1 (the weight of boundary loss), we try 1,5,10, and 25 where 5 brought the best performance.

![framework](https://user-images.githubusercontent.com/80099298/186313953-76343502-4e56-4e07-a09d-e7de5053f5cb.png)

### step1:
make images and labels to the suitable size.

### step2:
`python train_two.py` (two branch)
and you can change the batchsize, lr, name, dir, and epoch in config.txt

### Another  step:
`python train_one.py`  (one branch)


## To be Updated...
