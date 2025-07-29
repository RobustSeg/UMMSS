import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeSegmentation:
    def __init__(self, num_classes, feature_dim):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        # Initialize global prototypes for each class
        self.global_prototypes = torch.zeros((num_classes, feature_dim), requires_grad=True).to(
            'cuda')  # Set requires_grad=True

    def update_global_prototypes(self, current_prototypes):
        # Update global prototypes with current prototypes
        self.global_prototypes.data = 0.9 * self.global_prototypes.data + 0.1 * current_prototypes.data
        # Using `.data` avoids creating new computation graphs for the updated values.

    def compute_loss(self, features, labels):
        # Calculate per-class prototypes from current batch
        batch_prototypes = self.calculate_batch_prototypes(features, labels)

        # Update global prototypes
        self.update_global_prototypes(batch_prototypes)

        # Compute the prototype matching loss
        prototype_loss = self.prototype_loss(batch_prototypes)

        # Optionally, you can also compute segmentation loss here (e.g., CrossEntropyLoss)
        # segmentation_loss = F.cross_entropy(logits, labels)

        # Combine the losses
        total_loss = prototype_loss
        return total_loss

    def calculate_batch_prototypes(self, features, labels):
        # Initialize prototypes
        batch_prototypes = torch.zeros((self.num_classes, self.feature_dim), device=features.device)
        count = torch.zeros(self.num_classes, device=features.device)

        # Ensure labels are on the same device as features
        labels = labels.to(features.device)

        labels = labels.unsqueeze(1)
        # Resize labels to match feature map size
        labels_resized = F.interpolate(labels.float(), size=features.shape[2:], mode='nearest').long().squeeze(1)

        # Flatten features and resized labels
        b, c, h, w = features.size()
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels_resized = labels_resized.view(-1)
        # print("labels_resized:",labels_resized.shape)

        for i in range(self.num_classes):
            mask = (labels_resized == i)
            if mask.sum() > 0:
                batch_prototypes[i] = features[mask].mean(dim=0)
                count[i] = mask.sum()

        # Avoid division by zero
        count = count.clamp(min=1)
        return batch_prototypes / count.unsqueeze(1)

    def prototype_loss(self, batch_prototypes):
        # Calculate the loss between batch prototypes and global prototypes
        loss = F.mse_loss(batch_prototypes, self.global_prototypes)
        return loss

# # 假设我们有一个特征图和对应的标签
# features = torch.randn(2, 20, 256, 256).to('cuda')  # 特征大小为 (batch, channels, height, width)
# labels = torch.rand(2, 256, 256).to('cuda')  # 假设共有 5 个类别
#
# # 创建原型分割类
# proto_segmentation = PrototypeSegmentation(num_classes=5, feature_dim=20)
#
# prototype = proto_segmentation.calculate_batch_prototypes(features, labels)