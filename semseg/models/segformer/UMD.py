import torch
import torch.nn as nn
# from semseg.models.segformer.seg_block_UMDt import Seg as Seg_s
# from semseg.models.segformer.seg_block_select_UMD import Seg


def UMD(index: list, x_all: list, x_all_s: list):
    loss_umd = 0.0
    loss_kl = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
    B = len(x_all)
    for i in range(B):
        for j in range(4):
            for k in range(len(index)):
                # x_all[i][j][k, :, :, :] = torch.log_softmax(x_all[i][j][k, :, :, :], dim=1)
                # x_all_s[i][j][index[k], :, :, :] = torch.softmax(x_all_s[i][j][index[k], :, :, :], dim=1)
                #
                # loss_s = loss_kl(x_all[i][j][k, :, :, :], x_all_s[i][j][index[k], :, :, :]).clamp(min=0)
                # # loss_umd += loss_s
                # loss_umd = loss_umd + loss_s

                x_all_log_softmax = torch.log_softmax(x_all[i][j][k, :, :, :], dim=1)
                x_all_s_softmax = torch.softmax(x_all_s[i][j][index[k], :, :, :], dim=1)

                # 如果用教师的特征去替换学生的特征会损失多少特征，这个值应该越小越好
                loss_s = loss_kl(x_all_log_softmax, x_all_s_softmax).clamp(min=0)
                loss_umd += loss_s

    return loss_umd / B

#
# model = Seg("mit_b0", num_classes=19, pretrained=True)
# model_s = Seg_s("mit_b0", num_classes=19, pretrained=True)
#
# sample = [torch.zeros(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024),
#           torch.ones(2, 3, 1024, 1024)]
#
# logits, index, ms_feat = model(sample)
# print(ms_feat[0][0].shape)
# with torch.no_grad():
#     logits_s, ms_feat_s = model_s(sample)
# print(ms_feat_s[0][0].shape)
#
# loss = UMD(index, ms_feat, ms_feat_s)
# print(0)