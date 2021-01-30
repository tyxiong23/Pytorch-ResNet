import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import ResNet
import os
import os.path as path
from tensorboardX import SummaryWriter

#解析器设置 (包含超参数设置)
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--outf", default="model", help="folder to output images and model checkpoints")
parser.add_argument("--rec", default="record", help="folder to output training and testing records")
parser.add_argument("--print-freq", default=50, type=int, help="print frequency (default: 50)")
parser.add_argument("--save-checkpoints", default=10, type=int, help="save checkpoints at specified number of epoches")
parser.add_argument("--exp-id", default="1", help="record the id if experiment")
parser.add_argument("--layers", default=18, type=int, help="set the number of layers(default: 18)")
parser.add_argument("--epoches", default=200, type=int, help="set the number of epoches(default: 200)")
parser.add_argument("--batch-size", default=128, type=int, help="set the batch size of training(default: 128)")
parser.add_argument("--lr", default=0.1, type=float, help="set the intial learning rate(default: 0.1)")
parser.add_argument("--weight-decay", default=0.0, type=float, help="set the weight decay of optimizer(default: 0)")
parser.add_argument("--momentum", default=0.9, type=float, help="set the momentum of optimizer")
parser.add_argument("--cuda", default=0, type=int, help="choose which GPU to use in training")
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

#选择要训练的模型种类
try:
    Net = ResNet.__dict__["resnet{}".format(args.layers)]
except:
    print ("ResNet can only be chosen from {}.".format(ResNet.__all__))
    exit()

#对数据的预处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,    
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_set = torchvision.datasets.CIFAR10(root=path.join(path.dirname(__file__), './data'), train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root=path.join(path.dirname(__file__), './data'), train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

#CIFAR-10的标签
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

#定义ResNet模型
net = Net()
num_GPUs = torch.cuda.device_count()
print(num_GPUs, " GPU(s) are available!")
# if num_GPUs > 1:
#     net = nn.DataParallel(net)
net.to(device)
#多GPU并发处理

#损失函数和优化方式(训练过程中改变学习率)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150])

if __name__ == "__main__":
    save_dir = path.join(path.dirname(__file__), args.outf, args.exp_id)
    rec_dir = path.join(path.dirname(__file__), args.rec, args.exp_id)
    log_dir = path.join(rec_dir, "logs")
    if not path.exists(save_dir):
        os.makedirs(save_dir)
    if not path.exists(rec_dir):
        os.makedirs(rec_dir)
    if not path.exists(log_dir):
        os.makedirs(log_dir)
    max_accuracy = 0.0

    print(device)
    #使用tensorBoardX将网络结构画出来
    net_print = Net()
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(net_print, torch.rand(1, 3, 32, 32))

    with open(path.join(rec_dir, "acc{}.txt".format(args.layers)), "w") as fil_acc:
        with open(path.join(rec_dir, "log{}.txt".format(args.layers)), "w") as fil_log:
            print("XTY's ResNet{} starts training!".format(args.layers))
            print("\nepoches={}".format(args.epoches))
            print("batch_size={}".format(args.batch_size))
            print("weight_decay={}".format(args.weight_decay))
            fil_log.write("XTY's ResNet{} starts training!\n".format(args.layers))
            fil_log.write("\nepoches={}\n".format(args.epoches))
            fil_log.write("batch_size={}\n".format(args.batch_size))
            fil_log.write("weight_decay={}\n".format(args.weight_decay))
            fil_log.flush()
            for epoch in range(0, args.epoches):
                print("\nEpoch: %3d | @lr=%.3f @momentum=%.3g" 
                    % (epoch + 1, optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["momentum"]))
                fil_log.write("\nEpoch: %3d | @lr=%.3f @momentum=%.3g\n" 
                    % (epoch + 1, optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["momentum"]))
                net.train() #设置为训练模式
                if epoch % args.save_checkpoints == 0:
                    torch.save(net.state_dict(), path.join(save_dir, "Net%d_%03d.pth" % (args.layers, epoch + 1)))
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                test_average = 0
                for i, data in enumerate(train_loader, 0):
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if i % args.print_freq == 0:
                        print("[epoch:%3d, %3d/%3d] Loss: %.03f | Accuracy: %.3f%%"
                            % (epoch + 1, i + 1, length, sum_loss/(1 + i), 100.0 * correct / total))
                        fil_log.write("[epoch:%3d, %3d/%3d] Loss: %.03f | Accuracy: %.3f%%"
                            % (epoch + 1, i + 1, length, sum_loss/(1 + i), 100.0 * correct / total))
                        fil_log.write("\n")
                        fil_log.flush()
                
                train_avg = 100.0 * correct / total
                lr_scheduler.step()

                #每完成一个epoch,测试一次模型的准确率
                print("Tesing.....")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    net.eval() #设置为验证模式
                    for data in test_loader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    acc = 100.0 * correct / total
                    print("测试分类准确率为：%.3f%%" % acc)
                    print("Saving Model......")
                    if epoch % args.save_checkpoints == 0:
                        torch.save(net.state_dict(), path.join(path.dirname(__file__), args.outf, "Net50_%03d.pth" % (epoch + 1)))
                    fil_acc.write("Test: EPOCH=%3d, Accuracy=%.3f%%" % (epoch + 1, acc))
                    fil_acc.write("\n")
                    fil_acc.flush()
                    if acc > max_accuracy:
                        max_accuracy = acc
                        fil_best = open(path.join(rec_dir, "best_acc.txt"), "w")
                        fil_best.write("EPOCH=%3d, Best Accuracy=%.3f%%\n" % (epoch + 1, acc))
                        fil_best.close()

                    writer.add_scalar("Loss in training", loss, epoch + 1)
                    writer.add_scalars("Accuracy", {"test": acc, "max_test": max_accuracy, "train_avg": train_avg}, epoch + 1)
                    
    writer.close()
    torch.save(net.state_dict(), path.join(save_dir, "ResNet{}.pth".format(args.layers)))
    print("Training Finished, TotalEPOCH = %d. Well done!" % args.epoches)    