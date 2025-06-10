
import torch
import torch.nn as nn

def get_graph_feature(x, k, idx, device):
    batch_size = x.size(0)
    num_points = x.size(2)
    new_num_points = idx.size(1)
    # x = x.view(batch_size, -1, num_points)
    
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, new_num_points, k, num_dims) 
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    x = x.view(batch_size, num_points, 1, num_dims)[:,:new_num_points,:,:].repeat(1, 1, k, 1)
    
    # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = torch.cat((x, feature-x), dim=3).permute(0, 3, 1, 2).contiguous()
 
    return feature      # (batch_size, 2*num_dims, num_points, k)


def get_graph_feature_xpf(xf, k, idx, device):
    batch_size = xf.size(0)
    num_points = xf.size(2)
    
    # device = torch.device('cuda')

    new_num_points = idx.size(1)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    xf = xf.view(batch_size, -1, num_points)
    _, num_dims, _ = xf.size()

    xf = xf.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = xf.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, new_num_points, k, num_dims)

    # print( xf.view(batch_size, num_points, 1, num_dims).size() )

    xf = xf.view(batch_size, num_points, 1, num_dims)[:,:new_num_points,:,:].repeat(1, 1, k, 1)
    
    # feature = torch.cat((feature-xf, xf), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = feature-xf # TODO: CHECK WITHOUT
    feature = feature.permute(0, 3, 1, 2).contiguous()
  
    return feature       # (batch_size, num_dims, num_points, k)


class LottoNetDet(nn.Module):
    # def __init__(self, args, output_channels=40):
    def __init__(self, k, dropout, output_channels=3, ifs=6):
        super(LottoNetDet, self).__init__()
        self.k = k
        
        self.gn1 = nn.GroupNorm(32,64,eps=0.001)
        self.gn2 = nn.GroupNorm(32,128,eps=0.001)
        self.gn3 = nn.GroupNorm(32,128,eps=0.001)
        self.gn4 = nn.GroupNorm(32,128,eps=0.001)
        self.gn5 = nn.GroupNorm(32,128,eps=0.001)

        self.dps1 = nn.Dropout(p=dropout)
        self.dps2 = nn.Dropout(p=dropout)
        self.dpc1 = nn.Dropout(p=dropout)
        self.dpc2 = nn.Dropout(p=dropout)
        self.dpt1 = nn.Dropout(p=dropout)
        self.dpt2 = nn.Dropout(p=dropout)

        self.conv1 = nn.Sequential(nn.Conv2d(ifs, 64, kernel_size=1, bias=True),
                                   nn.ReLU(),
                                   self.gn1)
        self.conv2 = nn.Sequential(nn.Conv2d((64*2)+ifs, 128, kernel_size=1, bias=True),
                                   nn.ReLU(),
                                   self.gn2)
        self.conv3 = nn.Sequential(nn.Conv2d((128*2)+ifs, 128, kernel_size=1, bias=True),
                                   nn.ReLU(),
                                   self.gn3)
        self.conv4 = nn.Sequential(nn.Conv2d((128*2)+ifs, 128, kernel_size=1, bias=True),
                                   nn.ReLU(),
                                   self.gn4)
       
        self.conv5 = nn.Sequential(nn.Conv2d((128*2)+ifs, 128, kernel_size=1, bias=True),
                                   nn.ReLU(),
                                   self.gn5)
        
        
        self.seg1 = nn.Sequential(nn.Conv1d(2*128, 512, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.seg2 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.seg3 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.seg4 = nn.Conv1d(128, output_channels, kernel_size=1, bias=True)
        
        
        self.dist1 = nn.Sequential(nn.Conv1d(2*128+128, 512, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.dist2 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.dist3 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.dist4 = nn.Conv1d(128, 3, kernel_size=1, bias=True)
        
        self.top1 = nn.Sequential(nn.Conv1d(2*128+128, 512, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.top2 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.top3 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=True),
                                   nn.ReLU())
        self.top4 = nn.Conv1d(128, 3, kernel_size=1, bias=True)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.xavier_uniform_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, xp, idx_val, device):
        num_point = xp.size(2)

        dim2, dim3, dim4, dim5 = num_point//4, num_point//8, num_point//16, num_point//32

        idx1 = idx_val[:, :dim2, :]
        idx2 = idx_val[:, dim2:(dim2+dim3), :]
        idx3 = idx_val[:, (dim2+dim3):(dim2+dim3+dim4), :]
        idx4 = idx_val[:, (dim2+dim3+dim4):(dim2+dim3+dim4+dim5), :]
        idx5 = idx_val[:, (dim2+dim3+dim4+dim5):(dim2+dim3+dim4+dim5+dim5), :]
        dimlast = dim5

        x = get_graph_feature_xpf(xp, k=self.k, idx=idx1, device=device)
        
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k, idx=idx2, device=device)
        xg = get_graph_feature_xpf(xp, k=self.k, idx=idx2, device=device)
        x = torch.cat((xg, x), dim=1)

        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k, idx=idx3, device=device)
        xg = get_graph_feature_xpf(xp, k=self.k, idx=idx3, device=device)
        x = torch.cat((xg, x), dim=1)

        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k, idx=idx4, device=device)
        xg  = get_graph_feature_xpf(xp, k=self.k, idx=idx4, device=device)
        x = torch.cat((xg, x), dim=1)

        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x4, k=self.k, idx=idx5, device=device)
        xg  = get_graph_feature_xpf(xp, k=self.k, idx=idx5, device=device)
        x = torch.cat((xg, x), dim=1)
        
        x = self.conv5(x)
        x5 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((
                        x4[:,:,:dimlast], 
                        x5[:,:,:dimlast]
                        ), 
                        dim=1)
        
        x = self.seg1(x)
        x = self.seg2(x)
        x = self.dps1(x)
        xm = self.seg3(x)
        xm = self.dps2(xm)
        xc = self.seg4(xm)

        xn = torch.cat((
                        x4[:,:,:dimlast], 
                        x5[:,:,:dimlast], 
                        xm), 
                        dim=1)
        
        xrc = self.dist1(xn)
        xrc = self.dist2(xrc)
        xrc = self.dpc1(xrc)
        xrc = self.dist3(xrc)
        xrc = self.dpc2(xrc)
        xrc = self.dist4(xrc)

        xrt = self.top1(xn)
        xrt = self.top2(xrt)
        xrt = self.dpt1(xrt)
        xrt = self.top3(xrt)
        xrt = self.dpt2(xrt)
        xrt = self.top4(xrt)

        return xc, xrc, xrt
