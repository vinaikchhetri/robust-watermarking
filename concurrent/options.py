import argparse

def arg_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--algo", type=str, help="Run FedAvg")

    parser.add_argument("--K", type=int, help="total no. of clients.")
    parser.add_argument("--C", type=float, help="C-fraction of clients.")
    parser.add_argument("--E", type=int, help="local epochs.")
    parser.add_argument("--B", type=int, help="batch size.")
    parser.add_argument("--T", type=int, help="total no. of rounds.")
    parser.add_argument("--lr", type=float, help="learning rate.")

    parser.add_argument("--alpha", type=float, help="percentage of watermarks.")
    parser.add_argument("--target", type=str, help="random or single")
    parser.add_argument("--pattern", type=str, help="pixel or combination")

    parser.add_argument("--dataset", type=str, default = "mnist", help="dataset choice.")
    parser.add_argument("--iid", type=str, default = "true", help="data distribution.")
    parser.add_argument("--model", type=str, default = "nn", help="model choice.")
    parser.add_argument("--gpu", type=str, default = "cpu", help="gpu or cpu.")
    parser.add_argument("--name", type=str,  help="save stats list as ...")
    parser.add_argument("--num_attackers", type=int,  help="number of attackers")
    parser.add_argument("--num_classes", type=int,  help="number of classes")
    args = parser.parse_args()
    
    return args