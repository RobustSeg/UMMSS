import torch
import torch.nn as nn
from mmkd.Prototype import PrototypeSegmentation
# from semseg.models.segformer.seg_block_UMDt import Seg as Seg_s
# from semseg.models.segformer.seg_block_select_UMD import Seg


def PCUMD(x_all: list, x_all_t: list, lbl: torch.Tensor, prototype_all: list):
    loss_pumd = 0.0
    loss_mse = nn.MSELoss()
    loss_kl = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
    B = len(x_all)
    for i in range(B):
        batch_label = lbl[i].unsqueeze(0)
        # lbl shape: torch.Size([1, 1024, 1024])
        # print("lbl shape:",batch_label.shape)
        # print(batch_label.shape)
        for j in range(4):
            # x_all_t_feature = torch.mean(x_all[i][j],dim=0).unsqueeze(0)
            # x_all_t_prototype = prototype_all[j].calculate_batch_prototypes(x_all_t_feature, batch_label)
            # x_all_t_prototype_softmax = torch.softmax(x_all_t_prototype, dim=0)
            for k in range(x_all[i][j]):
                # x_all[i][j][k,:,:,:]维度为x_all torch.Size([32, 256, 256])
                # 为了迎合prototype类中的batch需要，此处unsqueeze(0)
                x_all_feature = x_all[i][j][k, :, :, :].unsqueeze(0)
                x_all_t_feature = x_all_t[i][j][k, :, :, :].unsqueeze(0)
                # x_all_log_softmax = torch.log_softmax(x_all[i][j][k, :, :, :], dim=1).unsqueeze(0)
                # x_all_s_softmax = torch.softmax(x_all_t[i][j][index[k], :, :, :], dim=1).unsqueeze(0)

                # print("x_all_log",x_all_log_softmax.shape)

                # prototype [Class number, dim]
                x_all_prototype = prototype_all[j].calculate_batch_prototypes(x_all_feature, batch_label)
                x_all_t_prototype = prototype_all[j].calculate_batch_prototypes(x_all_t_feature, batch_label)

                # print("prototype",x_all_s_softmax_prototype.shape)

                x_all_prototype_log_softmax = torch.log_softmax(x_all_prototype,dim=0)
                x_all_t_prototype_softmax = torch.softmax(x_all_t_prototype, dim=0)
                loss_pumd += loss_kl(x_all_prototype_log_softmax, x_all_t_prototype_softmax).clamp(min=0)
                # print(loss_pumd)

                # loss_pumd += loss_mse(x_all_prototype, x_all_t_prototype)


    return loss_pumd / B

#
# model = Seg("mit_b0", num_classes=25, pretrained=True)
# model_s = Seg_s("mit_b0", num_classes=25, pretrained=True)
#
# sample = [torch.zeros(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024),
#           torch.ones(2, 3, 1024, 1024)]
# lbl = torch.zeros(2, 1024, 1024)
#
# logits, index, ms_feat = model(sample)
# with torch.no_grad():
#     logits_s, ms_feat_s = model_s(sample)
# loss = PUMD(index, ms_feat, ms_feat_s, lbl, model.num_classes)
# print(0)