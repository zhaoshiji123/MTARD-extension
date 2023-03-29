import os
import argparse
import torch
from mtard_loss import *
from cifar10_models import *
import torchvision
from torchvision import transforms
from dataset import CIFAR10, CIFAR100, SVHN

# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

prefix = 'resnet18_mtard_cifar10_temp_entropy_f'
epochs = 300
batch_size = 128
epsilon = 8/255.0



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean, std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean, std),
])

trainset = CIFAR10(root='./data_cifar10', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = CIFAR10(root='./data_cifar10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

student = resnet18()
#student = torch.nn.DataParallel(student)
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

weight = {
    "adv_loss": 1/2.0,
    "nat_loss": 1/2.0,
}
init_loss_nat = None
init_loss_adv = None


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss

def entropy_value(a):
    value = torch.log(a+1e-5)*a
    return value

teacher = WideResNet()
teacher1_path = ''
state_dict = torch.load(teacher1_path)
teacher.load_state_dict(state_dict)
#teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
# teacher = teacher.half()
teacher.eval()


teacher_nat = resnet56()
teacher2_path = ''
state_dict_1 = torch.load(teacher2_path)
teacher_nat.load_state_dict(state_dict_1)
#teacher = torch.nn.DataParallel(teacher)
teacher_nat = teacher_nat.cuda()
teacher_nat.eval()


weight_learn_rate = 0.025
temp_learn_rate = 0.001

ce_loss = torch.nn.CrossEntropyLoss().cuda()
ce_loss_test = torch.nn.CrossEntropyLoss(reduction='none')
best_accuracy = 0

temp_adv = 1
temp_nat = 1

temp_max = 10
temp_min = 1

f = open('output_resnet18_mtard_cifar10_memory_teacher_b.txt', 'w')


for epoch in range(1,epochs+1):
    print('the {}th epoch '.format(epoch))
    print('the {}th epoch '.format(epoch),file=f)

    # temp_cicle_max = temp_min + min(epoch, 250) / 250 * (temp_max-temp_min)
    for step,(train_batch_data,train_batch_labels, index) in enumerate(trainloader):
        #print("step is :", step)
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        #weight_loss_optim.zero_grad()
        # with torch.no_grad():
        #     #teacher_logits = teacher(train_batch_data)
        #     teacher_nat_logits = teacher_nat(train_batch_data)
        student_adv_logits,teacher_adv_logits = rslad_inner_loss_ce_2(student,teacher,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)

        student.train()
        student_nat_logits = student(train_batch_data)
        with torch.no_grad():
            #teacher_logits = teacher(train_batch_data)
            teacher_nat_logits = teacher_nat(train_batch_data)

        kl_Loss1 = kl_loss(F.log_softmax(student_adv_logits,dim=1),F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1))
        kl_Loss2 = kl_loss(F.log_softmax(student_nat_logits,dim=1),F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1))

        kl_Loss1 = torch.mean(kl_Loss1)
        kl_Loss2 = torch.mean(kl_Loss2)

        adv_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1)))
        nat_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1)))

        temp_adv = temp_adv - temp_learn_rate * torch.sign((adv_teacher_entropy.detach() / nat_teacher_entropy.detach() - 1)).item()
        temp_nat = temp_nat - temp_learn_rate * torch.sign((nat_teacher_entropy.detach() / adv_teacher_entropy.detach() - 1)).item()
        temp_adv = max(min(temp_max, temp_adv), temp_min)
        temp_nat = max(min(temp_max, temp_nat), temp_min)


        if init_loss_nat == None:
            init_loss_nat = kl_Loss2.item()
        if init_loss_adv == None:
            init_loss_adv = kl_Loss1.item()


        G_avg = (kl_Loss1.item() + kl_Loss2.item()) / len(weight)
        lhat_adv = kl_Loss1.item() / init_loss_adv
        lhat_nat = kl_Loss2.item() / init_loss_nat
        lhat_avg = (lhat_adv + lhat_nat) / len(weight)


        inv_rate_adv = lhat_adv / lhat_avg
        inv_rate_nat = lhat_nat / lhat_avg


        weight["nat_loss"] = weight["nat_loss"] - weight_learn_rate *(weight["nat_loss"] - inv_rate_nat/(inv_rate_adv + inv_rate_nat))
        weight["adv_loss"] = weight["adv_loss"] - weight_learn_rate *(weight["adv_loss"] - inv_rate_adv/(inv_rate_adv + inv_rate_nat))


        num_losses = len(weight)

        if weight["adv_loss"] <0:
            weight["adv_loss"] = 0
        if weight["nat_loss"]< 0:
            weight["nat_loss"] = 0

        coef = 1.0/(weight["adv_loss"] + weight["nat_loss"])
        weight["adv_loss"] *= coef
        weight["nat_loss"] *= coef

        #the model update, loss has been updated above
        total_loss = weight["adv_loss"]*kl_Loss1 + weight["nat_loss"]*kl_Loss2

        total_loss.backward()
        optimizer.step()
        if step%100 == 0:
            print('temp_adv ', temp_adv,'temp_nat ', temp_nat)
            print('weight_nat ', weight["nat_loss"],'nat_loss ',kl_Loss2.item(),' weight_adv ', weight["adv_loss"],' adv_loss ',kl_Loss1.item())
            # assert False
            print('temp_adv ', temp_adv,'temp_nat ', temp_nat,file=f)
            print('weight_nat ', weight["nat_loss"],'nat_loss ',kl_Loss2.item(),' weight_adv ', weight["adv_loss"],' adv_loss ',kl_Loss1.item(),file=f)

    if (epoch % 20 == 0 and epoch <215) or (epoch%1 == 0 and epoch >= 215) or epoch == 1:
        loss_nat_test = AverageMeter()
        loss_adv_test = AverageMeter()

        student.eval()
        optimizer.zero_grad()
        test_accs = []
        test_accs_naturals = []
        for step,(test_batch_data,test_batch_labels,index) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            with torch.no_grad():
                logits = student(test_ifgsm_data)
                loss = ce_loss(logits, test_batch_labels)
            loss = loss.float()
            # measure accuracy and record loss
            loss_adv_test.update(loss.item(), test_batch_data.size(0))
            
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
            # break
        test_accs = np.array(test_accs)
        test_adv = np.sum(test_accs==0)/len(test_accs)
        print('robust acc ',np.sum(test_accs==0)/len(test_accs),file=f)
        print('robust acc loss {loss.val:.4f}'.format(loss = loss_adv_test),file=f)
        for step,(test_batch_data,test_batch_labels,index) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            with torch.no_grad():
                logits = student(test_batch_data)
                loss = ce_loss(logits, test_batch_labels)
            loss = loss.float()
            # measure accuracy and record loss
            loss_nat_test.update(loss.item(), test_batch_data.size(0))
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs_naturals = test_accs_naturals + predictions.tolist()
            # break
        test_accs_naturals = np.array(test_accs_naturals)
        test_nat = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
        print('natural acc ',np.sum(test_accs_naturals==0)/len(test_accs_naturals),file=f)
        print('natural acc loss {loss.val:.4f}'.format(loss = loss_nat_test),file=f)
        if (test_nat + test_adv) / 2 > best_accuracy:
            state = { 'model': student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./resnet18_mtard_cifar10_temp_entropy_f/' + prefix + "_best"+ '.pth')
            best_accuracy = (test_nat + test_adv) /2
            print("best accuracy:",str(best_accuracy),file=f)
        if epoch % 20 == 0:
            state = { 'model': student.state_dict(),
                    'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./resnet18_mtard_cifar10_temp_entropy_f/' + prefix + str(test_adv) +"_"+ str(test_nat) +"_"+str(epoch)+ '.pth')
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        weight_learn_rate *= 0.1
        temp_learn_rate *= 0.1