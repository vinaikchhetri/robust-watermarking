#!/bin/sh

#python3 main.py --algo="FedAvg" --K=1 --C=1 --E=100 --B=1024 --T=1 --lr=0.1 --alpha=0.05 --target="all" --pattern="combination" --gpu="gpu" --model="cnn" --name="exp_x" > ../logs/exp_x.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.1 --gpu="gpu" --model="nn" --name="exp_1" > ../logs/exp_1.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.2 --gpu="gpu" --model="nn" --name="exp_2" > ../logs/exp_2.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.3 --gpu="gpu" --model="nn" --name="exp_3" > ../logs/exp_3.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.4 --gpu="gpu" --model="nn" --name="exp_4" > ../logs/exp_4.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.6 --gpu="gpu" --model="nn" --name="exp_5" > ../logs/exp_5.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.7 --gpu="gpu" --model="nn" --name="exp_6" > ../logs/exp_6.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.8 --gpu="gpu" --model="nn" --name="exp_7" > ../logs/exp_7.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=0.9 --gpu="gpu" --model="nn" --name="exp_8" > ../logs/exp_8.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_9" > ../logs/exp_9.txt


# python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_10" > ../logs/exp_10.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_11" > ../logs/exp_11.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.3 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_12" > ../logs/exp_12.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.4 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_13" > ../logs/exp_13.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_14" > ../logs/exp_14.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.6 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_15" > ../logs/exp_15.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.7 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_16" > ../logs/exp_16.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.8 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_17" > ../logs/exp_17.txt

# python3 main.py --algo="FedAvg" --K=100 --C=0.9 --E=5 --B=50 --T=500 --lr=0.01 --alpha=1 --gpu="gpu" --model="nn" --name="exp_18" > ../logs/exp_18.txt



# mnist
# attacker = 0
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="cnn" --name="mnist-0" --num_attackers=0 --target="single" --num_classes=10 --dataset="mnist" 
# attacker = 1
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="cnn" --name="mnist-1" --num_attackers=1 --target="single" --num_classes=10 --dataset="mnist" 

# attacker = 10
#python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="cnn" --name="mnist-10" --num_attackers=10 --target="single" --num_classes=10 --dataset="mnist" 

# attacker = 25
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="cnn" --name="mnist-25" --num_attackers=25 --target="single" --num_classes=10 --dataset="mnist" 

# attacker = 49
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="cnn" --name="mnist-49" --num_attackers=49 --target="single" --num_classes=10 --dataset="mnist" 

# cifar-10
# attacker = 0
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-10-0" --num_attackers=0 --target="single" --num_classes=10 --dataset="cifar-10" 

# attacker = 1
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-10-1" --num_attackers=1 --target="single" --num_classes=10 --dataset="cifar-10" 

# attacker = 10
#python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-10-10" --num_attackers=10 --target="single" --num_classes=10 --dataset="cifar-10" 

# attacker = 25
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-10-25" --num_attackers=25 --target="single" --num_classes=10 --dataset="cifar-10" 

# attacker = 49
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.1 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-10-49" --num_attackers=49 --target="single" --num_classes=10 --dataset="cifar-10" 

# cifar-100
# attacker = 0
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.01 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-100-0" --num_attackers=0 --target="single" --num_classes=100 --dataset="cifar-100" 
# attacker = 1
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.01 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-100-1" --num_attackers=1 --target="single" --num_classes=100 --dataset="cifar-100" 

# attacker = 10
#python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.01 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-100-10" --num_attackers=10 --target="single" --num_classes=100 --dataset="cifar-100" 

# attacker = 25
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.01 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-100-25" --num_attackers=25 --target="single" --num_classes=100 --dataset="cifar-100" 

# attacker = 49
python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=10 --B=64 --T=30 --lr=0.01 --alpha=0.05 --pattern="combination" --gpu="gpu" --model="resnet" --name="cifar-100-49" --num_attackers=49 --target="single" --num_classes=100 --dataset="cifar-100" 
