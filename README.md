## A Zeroth-Order Adaptive Frank-Wolfe Algorithm for Resource Allocation in Internet of Things: Convergence Analysis

Source Code for "A Zeroth-Order Adaptive Frank-Wolfe Algorithm for Resource Allocation in Internet of Things: Convergence Analysis".

We adopt pytorch to run Robust Black-box Binary Classification and Black-box Adversarial Attack experiments, use MATLAB to run sensor selection experiment.

### Prerequisites

The environment configurations for Robust Black-box Binary Classification and Black-box Adversarial Attack are specified in the environment.ymal file.

Sensor selection experiment is performed using MATLAB-2016a.

### For Robust Black-box Binary Classification

The source code from https://github.com/TLMichael/Acc-SZOFW.

#### Download dataset manually

mkdir -p ~/datasets/phishing/

mkdir -p ~/datasets/a9a/

mkdir -p ~/datasets/w8a/

mkdir -p ~/datasets/covtype/

wget -P ~/datasets/phishing/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing

wget -P ~/datasets/a9a/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a

wget -P ~/datasets/w8a/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a

wget -P ~/datasets/covtype/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2

#### Run

cd app

python run.py

#### Result

The result are save in ./app/results/tsdata/ZO_AdaSFW/CooGE/ada.xlsx

### For Black-box Adversarial Attack

The source code from https://github.com/FedericoZanotti/Zeroth-Order-Methods-for-Adversarial-Machine-Learning

#### RUN

All code is in ./content/Optimization_Project_2020_2021 (1).py

There are 10 algorithms. Select algorithm to run

    loss_Z, x_Z,_=ZSCG(epochs, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)

    loss_FZ, x_FZ,_=FZFW(epochs, 784,n, 0.1,10**-1,10**-3 ,x, y_true_in, verbose=True)

    loss_ZOAda, ZOAda,_=ZO_AdaSFW(epochs, 2, 784,n, 0.1,10**-1,10**-3 ,x, y_true_in, verbose=True)

    loss_ZOSFW, x_Z,_=ZO_SFW(epochs, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)

    loss_Acc_SZOFW_U, x_Z,_=Acc_SZOFW_Uni(epochs, n, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)

    loss_Acc_SZOFW_C, x_Z,_=Acc_SZOFW_Coo(epochs, n, 784, 0.1, -1, x, y_true_in,10**-5,10**-1, verbose=True)

    loss_Acc_SZOFWX_U, x_Z,_=Acc_SZOFWX_Uni(epochs, n, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)

    loss_Acc_SZOFWX_C, x_Z,_=Acc_SZOFWX_Coo(epochs, n, 784, 0.1, -1, x, y_true_in,10**-5,10**-1, verbose=True)

    loss_SFW_Grad, x_Z,_=SFW_Grad(epochs, 784, 0.1, 30, x, y_true_in,10**-5,10**-1, verbose=True)


"""

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer()) 
 
        dnn.load_weights(path)
 
        loss_AdaSFW = AdaSFW(epochs, 2,784,n, 0.1,10**-1 ,x, y_true_in, verbose=True,sess=sess)
 
"""

#### Result

The result are save in ./content/results/

### Sensor Selection Experiments

The source code from https://github.com/lsjxjtu/ZOO-ADMM

#### Run

cd Sensor

run Main_ZOOADMM_App_SensrSel.m
