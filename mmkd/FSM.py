import torch
import torch.nn as nn
from torch.nn import functional as F

cmnext_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}

class PredictorConv(nn.Module):
    def __init__(self, embed_dim=32, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_

class TokenSelectionModule(nn.Module):
    def __init__(self, embed_dim=32, num_modals=2):
        super().__init__()
        self.num_modals = num_modals
        self.extra_score_predictor = PredictorConv(embed_dim, self.num_modals)
    def forward(self, x: list):
        x_  = self.tokenselect(x)
        return x_

    def tokenselect(self, x_ext):
        # 1. 计算每个特征的注意力分数
        x_scores = self.extra_score_predictor(x_ext)

        # 2. 创建加权特征（不修改原始x_ext）
        weighted_features = []
        for i in range(len(x_ext)):
            # 计算加权特征但不修改原始值
            weighted_feature = x_scores[i] * x_ext[i] + x_ext[i]
            weighted_features.append(weighted_feature)

        # 3. 堆叠所有加权特征并找到最大值索引（遮罩）
        stacked = torch.stack(weighted_features)  # [num_features, B, C, H, W]
        max_indices = torch.argmax(stacked, dim=0)  # [B, C, H, W] 每个位置的最大值索引

        # 4. 使用遮罩从原始特征中选择值
        x_f = torch.zeros_like(x_ext[0])  # 创建输出容器
        total_pixels = torch.tensor(max_indices.numel(), dtype=torch.float32)  # 总像素数

        # 5. 计算每个模态被选择的百分比
        percentages = []
        for i in range(len(x_ext)):
            # 为当前特征图创建遮罩
            mask = (max_indices == i)
            # 计算当前模态被选择的像素数量
            selected_pixels = torch.sum(mask).float()
            # 计算百分比
            percentage = (selected_pixels / total_pixels).item()
            percentages.append(percentage)

            # 将遮罩应用于原始特征（未加权）
            x_f[mask] = x_ext[i][mask]

        return x_f, percentages

    # def tokenselect(self, x_ext):
    #     B, C, H, W = x_ext[0].shape
    #
    #     # 1. 获取每个模态的打分（未归一化）
    #     raw_scores = self.extra_score_predictor(x_ext)  # list of [B, 1, H, W]
    #
    #     # 2. 堆叠并 softmax（温度调节）
    #     scores_tensor = torch.stack(raw_scores, dim=0)  # [num_modals, B, 1, H, W]
    #     normed_scores = F.softmax(scores_tensor / 0.2, dim=0)  # [num_modals, B, 1, H, W]
    #
    #     # 3. 使用 softmax 权重加权模态特征
    #     x_ext_stack = torch.stack(x_ext, dim=0)  # [num_modals, B, C, H, W]
    #     weights = normed_scores.expand(-1, -1, C, -1, -1)  # 匹配通道维度
    #     weighted_features = weights * x_ext_stack  # [num_modals, B, C, H, W]
    #
    #     fused = torch.sum(weighted_features, dim=0)  # [B, C, H, W]
    #
    #     # 4. 计算每个模态的被选比例（用 argmax）
    #     max_indices = torch.argmax(normed_scores, dim=0).squeeze(1)  # [B, H, W]
    #     total_pixels = max_indices.numel()
    #     percentages = []
    #     for i in range(self.num_modals):
    #         count = torch.sum(max_indices == i).float()
    #         percentages.append((count / total_pixels).item())
    #
    #     return fused, percentages

# 确保包含原始代码中的类定义（PredictorConv 和 TokenSelectionModule）

class TokenSelectionMultiModule(nn.Module):
    def __init__(self, backbone:str ="B0", num_modals=4):
        super().__init__()
        embed_dims, depths = cmnext_settings[backbone]
        print(embed_dims)
        self.tokenselect1 = TokenSelectionModule(embed_dim=embed_dims[0],num_modals=num_modals)
        self.tokenselect2 = TokenSelectionModule(embed_dim=embed_dims[1],num_modals=num_modals)
        self.tokenselect3 = TokenSelectionModule(embed_dim=embed_dims[2],num_modals=num_modals)
        self.tokenselect4 = TokenSelectionModule(embed_dim=embed_dims[3],num_modals=num_modals)
    def forward(self, x: list):
        x_0, p1 = self.tokenselect1.tokenselect([x[0][0],x[1][0]])
        x_1, p2 = self.tokenselect2.tokenselect([x[0][1],x[1][1]])
        x_2, p3 = self.tokenselect3.tokenselect([x[0][2],x[1][2]])
        x_3, p4 = self.tokenselect4.tokenselect([x[0][3],x[1][3]])
        x_ = [x_0,x_1,x_2,x_3]
        p = (p1[0] + p2[0] + p3[0] + p4[0])/4
        return x_, [p, 1-p]

def test_token_selection_with_percentage():
    # 1. 初始化模块
    embed_dim = 64
    num_modals = 2
    model = TokenSelectionModule(embed_dim, num_modals)

    # 2. 创建测试输入
    input_tensors = [
        torch.randn(2, embed_dim, 128, 128),  # 模态1
        torch.randn(2, embed_dim, 128, 128)  # 模态2
    ]

    # 3. 前向传播
    output, percentages = model(input_tensors)

    # 4. 验证输出张量
    assert isinstance(output, torch.Tensor), "输出应该是张量"
    assert output.shape == (2, embed_dim, 128, 128), f"形状错误，预期 [2, {embed_dim}, 256, 256]，实际 {output.shape}"

    # 5. 验证百分比输出
    assert isinstance(percentages, list), "百分比输出应该是列表"
    assert len(percentages) == num_modals, f"应该有 {num_modals} 个百分比值"

    # 验证百分比在0-1范围内
    for p in percentages:
        assert 0.0 <= p <= 1.0, f"百分比值 {p} 超出范围"
        print(p)

    # 验证百分比总和接近1.0
    total_percentage = sum(percentages)
    assert abs(total_percentage - 1.0) < 1e-5, f"百分比总和应为1.0，实际是 {total_percentage}"

    # # 6. 验证百分比计算逻辑
    # scores = model.extra_score_predictor(input_tensors)
    # stacked = torch.stack([s * inp + inp for s, inp in zip(scores, input_tensors)])
    # max_indices = torch.argmax(stacked, dim=0)
    #
    # # 计算实际百分比
    # total_pixels = max_indices.numel()
    # actual_percentages = []
    # for i in range(num_modals):
    #     mask = (max_indices == i)
    #     actual_percentages.append(torch.sum(mask).item() / total_pixels)
    #
    # # 比较计算值
    # for i, (p, ap) in enumerate(zip(percentages, actual_percentages)):
    #     assert abs(p - ap) < 1e-5, f"模态 {i} 百分比计算错误: 预期 {ap}，实际 {p}"
    #
    # print("测试通过！")
    # print(f"输出形状: {output.shape}")
    # print(f"模态百分比: {percentages}")
    # print(f"百分比总和: {sum(percentages):.4f}")


# 运行测试
if __name__ == "__main__":
    test_token_selection_with_percentage()