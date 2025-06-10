import numpy as np
from dataloaders import PCLoader
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import LottoNetDet
from torch.utils.data import DataLoader
from util import ClassicLoss, IOStream
import sklearn.metrics as metrics

import multiprocessing

            
def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_train.py outputs' + '/'+args.exp_name + '/' + 'main_train.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp dataloaders.py outputs' + '/' + args.exp_name + '/' + 'dataloaders.py.backup')
    
def worker_init_fn(*_):
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OMP_NUM_THREADS'] = '1'
    pass

def load_data(loader):
    for batch in loader:
        pass  # Simulate processing

def train(args, io):
 
    val_loader = DataLoader(PCLoader(partition='test', num_points=args.num_points, nn=args.k, dataset_index=0),
                          num_workers=16, batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                          multiprocessing_context=multiprocessing.get_context("spawn"))

    device = torch.device("cuda" if args.cuda else "cpu")

    number_of_instances = 3

    model = LottoNetDet(k=args.k, dropout=args.dropout, ifs=6, output_channels=number_of_instances).to(device)
    model = nn.DataParallel(model)
    if args.model_root != '':
        model.load_state_dict(torch.load(args.model_root))

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0004)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    scheduler = StepLR(opt, 20, 0.5)

    fl = ClassicLoss(gammas=[0.3891, 1.7064, 0.9045]) 
    criterion = fl.forward

    dataloader_list = []
    
    for i in range(1,50):
        dataloader_list.append( PCLoader(partition='train', num_points=args.num_points, nn=args.k, dataset_index=i) )

    train_loader = DataLoader(torch.utils.data.ConcatDataset(dataloader_list), 
                          num_workers=10, batch_size=args.batch_size, shuffle=False, drop_last=True,
                          worker_init_fn=worker_init_fn,
                          persistent_workers=True,
                          multiprocessing_context=multiprocessing.get_context("spawn"))

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        train_cen_dist = []
        train_top_dist = []

        train_cen_dist_list = [[],[],[]]
        train_top_dist_list = [[],[],[]]

        for data, seg, cen, top, idx, vis in train_loader:
            data, seg, cen, top, idx, vis = data.to(device), seg.to(device), cen.to(device), top.to(device), idx.to(device), vis.to(device)
            
            data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred, cen_pred, top_pred = model(data, idx, device)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            cen_pred = cen_pred.permute(0, 2, 1).contiguous()
            top_pred = top_pred.permute(0, 2, 1).contiguous()
            
            loss = criterion(seg_pred.view(-1, number_of_instances), seg.view(-1,1).squeeze(), cen_pred.view(-1, 3), cen.view(-1,3).squeeze(), top_pred.view(-1, 3), top.view(-1,3).squeeze(), vis.view(-1,1).squeeze())

            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            train_loss += loss.item() * batch_size


            seg_np = seg.view(-1,1).squeeze().cpu().numpy()
            seg_obj = seg_np > 0

            cen_pred_np = cen_pred.view(-1,3).detach().cpu().numpy()
            cen_np = cen.view(-1,3).cpu().numpy()
            top_pred_np = top_pred.view(-1,3).detach().cpu().numpy()
            top_np = top.view(-1,3).cpu().numpy()
                   
            cen_pred_np = cen_pred_np[seg_obj,:]
            cen_np = cen_np[seg_obj,:]
            top_pred_np = top_pred_np[seg_obj,:]
            top_np = top_np[seg_obj,:]
            
            train_cen_dist_temp = np.sum(np.sqrt(np.sum((cen_pred_np-cen_np)**2,axis=1)))/cen_np.shape[0]
            train_top_dist_temp = np.sum(np.sqrt(np.sum((top_pred_np-top_np)**2,axis=1)))/top_np.shape[0]

            train_cen_dist.append( train_cen_dist_temp )
            train_top_dist.append( train_top_dist_temp )

            for obj_idx in range(number_of_instances):
                seg_obj = seg_np == obj_idx
                
                cen_pred_np = cen_pred.view(-1,3).detach().cpu().numpy()
                cen_np = cen.view(-1,3).cpu().numpy()
                top_pred_np = top_pred.view(-1,3).detach().cpu().numpy()
                top_np = top.view(-1,3).cpu().numpy()
                
                cen_pred_np_obj = cen_pred_np[seg_obj,:]
                cen_np_obj = cen_np[seg_obj,:]
                top_pred_np_obj = top_pred_np[seg_obj,:]
                top_np_obj = top_np[seg_obj,:]

                train_cen_dist_temp = np.sum(np.sqrt(np.sum((cen_pred_np_obj-cen_np_obj)**2,axis=1)))/cen_np_obj.shape[0] if cen_np_obj.shape[0] > 0 else 0
                train_top_dist_temp = np.sum(np.sqrt(np.sum((top_pred_np_obj-top_np_obj)**2,axis=1)))/top_np_obj.shape[0] if top_np_obj.shape[0] > 0 else 0

                train_cen_dist_list[obj_idx].append( train_cen_dist_temp )
                train_top_dist_list[obj_idx].append( train_top_dist_temp )


            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            train_true_cls.append(seg_np.reshape(-1))
            train_pred_cls.append(pred_np.reshape(-1))
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_cen_dist = np.mean(train_cen_dist)
        train_top_dist = np.mean(train_top_dist)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        outstr = 'Train %d, lr: %.6f, loss: %.6f, train acc: %.6f, train avg acc: %.6f, cen dist: %.6f, top dist: %.6f' % (epoch,
                                                                                  opt.param_groups[0]['lr'],
                                                                                  train_loss*1.0/count,
                                                                                  train_acc,
                                                                                  avg_per_class_acc,
                                                                                  train_cen_dist,
                                                                                  train_top_dist)
        io.cprint(outstr)
        for obj_idx in range(number_of_instances):
            io.cprint( "Train Obj %d distances cen: %.6f top: %.6f" % (obj_idx, np.mean(train_cen_dist_list[obj_idx]), np.mean(train_top_dist_list[obj_idx]) ))
        if epoch%10 == 0:
            io.cprint( metrics.classification_report(train_true_cls, train_pred_cls) )

        ####################
        # Validation
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
       
        test_cen_dist = []
        test_top_dist = []
        
        ind_cen_dist = [[],[],[]]
        ind_top_dist = [[],[],[]]


        for data, seg, cen, top, idx, vis in val_loader:
            data, seg, cen, top, idx, vis = data.to(device), seg.to(device), cen.to(device), top.to(device), idx.to(device), vis.to(device)
            data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred, cen_pred, top_pred = model(data, idx, device)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            cen_pred = cen_pred.permute(0, 2, 1).contiguous()
            top_pred = top_pred.permute(0, 2, 1).contiguous()

            loss = criterion(seg_pred.view(-1, number_of_instances), seg.view(-1,1).squeeze(), cen_pred.view(-1, 3), cen.view(-1,3).squeeze(), top_pred.view(-1, 3), top.view(-1,3).squeeze(), vis.view(-1,1).squeeze())

            cen_pred_np = cen_pred.view(-1,3).detach().cpu().numpy()
            cen_np = cen.view(-1,3).cpu().numpy()

            top_pred_np = top_pred.view(-1,3).detach().cpu().numpy()
            top_np = top.view(-1,3).cpu().numpy()

            seg_np = seg.view(-1,1).squeeze().cpu().numpy()
            seg_obj = seg_np > 0

            cen_pred_np = cen_pred_np[seg_obj,:]
            cen_np = cen_np[seg_obj,:]
            top_pred_np = top_pred_np[seg_obj,:]
            top_np = top_np[seg_obj,:]
       
            test_cen_dist_temp = np.sum(np.sqrt(np.sum((cen_pred_np-cen_np)**2,axis=1)))/cen_np.shape[0]
            test_top_dist_temp = np.sum(np.sqrt(np.sum((top_pred_np-top_np)**2,axis=1)))/top_np.shape[0]

            test_cen_dist.append( test_cen_dist_temp )
            test_top_dist.append( test_top_dist_temp )

            for obj_idx in range(number_of_instances):
                seg_np = seg.view(-1,1).squeeze().cpu().numpy()
                seg_obj = seg_np == obj_idx
                cen_pred_np = cen_pred.view(-1,3).detach().cpu().numpy()
                cen_np = cen.view(-1,3).cpu().numpy()

                top_pred_np = top_pred.view(-1,3).detach().cpu().numpy()
                top_np = top.view(-1,3).cpu().numpy()

                cen_pred_np = cen_pred_np[seg_obj,:]
                cen_np = cen_np[seg_obj,:]
                top_pred_np = top_pred_np[seg_obj,:]
                top_np = top_np[seg_obj,:]
                
                test_cen_dist_temp = np.sum(np.sqrt(np.sum((cen_pred_np-cen_np)**2,axis=1)))/cen_np.shape[0]
                test_top_dist_temp = np.sum(np.sqrt(np.sum((top_pred_np-top_np)**2,axis=1)))/top_np.shape[0]

                ind_cen_dist[obj_idx].append( test_cen_dist_temp ) 
                ind_top_dist[obj_idx].append( test_top_dist_temp ) 

            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_cen_dist = np.mean(test_cen_dist)
        test_top_dist = np.mean(test_top_dist)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        outstr = 'Val %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, center dist: %.6f, top dist: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              test_cen_dist,
                                                                              test_top_dist)
        io.cprint(outstr)
        for obj_idx in range(number_of_instances):
            io.cprint( "Val Obj %d distances cen: %.6f top: %.6f" % (obj_idx, np.mean(ind_cen_dist[obj_idx]), np.mean(ind_top_dist[obj_idx]) ))
        if epoch%10 == 0:
            io.cprint( metrics.classification_report(test_true_cls, test_pred_cls) )

        torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)

    torch.save(model.state_dict(), 'outputs/%s/models/model_final.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(PCLoader(partition='test', num_points=args.num_points, dataset_index=3, veclen = 50, 
                          data_path="/workspace/bop/icbin/test/", image_ext=".png", 
                          camera_param_path="/workspace/bop/icbin/test/000001/scene_camera.json",),
                          num_workers=10, batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                          multiprocessing_context=multiprocessing.get_context("spawn"))
    device = torch.device("cuda" if args.cuda else "cpu")
    
    number_of_instances = 3

    model = LottoNetDet(k=args.k, dropout=args.dropout, ifs=6, output_channels=number_of_instances).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_root))
    model = model.eval()

    print(str(model))
    ####################
    # Test
    ####################
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
   
    test_cen_dist = []
    test_top_dist = []
        
    ind_cen_dist = [[],[],[]]
    ind_top_dist = [[],[],[]]
    
    for data, seg, cen, top, idx, vis in test_loader:
        data, seg, cen, top, idx = data.to(device), seg.to(device), cen.to(device), top.to(device), idx.to(device)
        data = data.permute(0, 2, 1)

        batch_size = data.size()[0]
        seg_pred, cen_pred, top_pred = model(data, idx, device)
        
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        cen_pred = cen_pred.permute(0, 2, 1).contiguous()
        top_pred = top_pred.permute(0, 2, 1).contiguous()

        cen_pred_np = cen_pred.view(-1,3).detach().cpu().numpy()
        cen_np = cen.view(-1,3).cpu().numpy()

        top_pred_np = top_pred.view(-1,3).detach().cpu().numpy()
        top_np = top.view(-1,3).cpu().numpy()
        
        seg_np = seg.view(-1,1).squeeze().cpu().numpy()
        seg_obj = seg_np > 0

        cen_pred_np = cen_pred_np[seg_obj,:]
        cen_np = cen_np[seg_obj,:]
        top_pred_np = top_pred_np[seg_obj,:]
        top_np = top_np[seg_obj,:]
        
        test_cen_dist_temp = np.sum(np.sqrt(np.sum((cen_pred_np-cen_np)**2,axis=1)))/cen_np.shape[0]
        test_top_dist_temp = np.sum(np.sqrt(np.sum((top_pred_np-top_np)**2,axis=1)))/top_np.shape[0]

        test_cen_dist.append(test_cen_dist_temp)
        test_top_dist.append(test_top_dist_temp)

        for obj_idx in range(number_of_instances):
            seg_np = seg.view(-1,1).squeeze().cpu().numpy()
            seg_obj = seg_np == obj_idx
            cen_pred_np = cen_pred.view(-1,3).detach().cpu().numpy()
            cen_np = cen.view(-1,3).cpu().numpy()

            top_pred_np = top_pred.view(-1,3).detach().cpu().numpy()
            top_np = top.view(-1,3).cpu().numpy()

            cen_pred_np = cen_pred_np[seg_obj,:]
            cen_np = cen_np[seg_obj,:]
            top_pred_np = top_pred_np[seg_obj,:]
            top_np = top_np[seg_obj,:]
            
            test_cen_dist_temp = np.sum(np.sqrt(np.sum((cen_pred_np-cen_np)**2,axis=1)))/cen_np.shape[0]
            test_top_dist_temp = np.sum(np.sqrt(np.sum((top_pred_np-top_np)**2,axis=1)))/top_np.shape[0]

            ind_cen_dist[obj_idx].append( test_cen_dist_temp ) 
            ind_top_dist[obj_idx].append( test_top_dist_temp ) 
        

        pred = seg_pred.max(dim=2)[1]
        count += batch_size
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_cen_dist = np.mean(test_cen_dist)
    test_top_dist = np.mean(test_top_dist)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    outstr = 'test acc: %.6f, test avg acc: %.6f, center dist: %.6f, top dist: %.6f' % (
                                                                          test_acc,
                                                                          avg_per_class_acc,
                                                                          test_cen_dist,
                                                                          test_top_dist)
    io.cprint(outstr)
    for obj_idx in range(number_of_instances):
        io.cprint( "Val Obj %d distances cen: %.6f top: %.6f" % (obj_idx, np.mean(ind_cen_dist[obj_idx]), np.mean(ind_top_dist[obj_idx]) ))
    io.cprint( metrics.classification_report(test_true_cls, test_pred_cls) )



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='step', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=65536,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
