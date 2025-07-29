import torch
import torch.nn as nn

def UMD(index: list, x_all: list, x_all_s: list):

    B = len(x_all)  # Assuming a batch size of 2
    x_all_feat = []
    x_all_s_feat = []
    for i in range(B):
        for j in range(4):
            for k in range(len(index)):
                # Check if index[k] is within valid range for x_all_s
                if index[k] >= x_all_s[i][j].shape[0]:
                    raise IndexError(f"Index {index[k]} out of bounds for tensor of shape {x_all_s[i][j].shape}")
                
                # Get the log-softmax and softmax of the tensors
                x_all_i_j_k_log = torch.log_softmax(x_all[i][j][k,:,:,:], dim=1)
                x_all_s_i_j_k = torch.softmax(x_all_s[i][j][index[k],:,:,:], dim=1)

                # Check for shape compatibility
                assert x_all[i][j][k,:,:,:].shape == x_all_s[i][j][index[k],:,:,:].shape, "Shape mismatch between x_all and x_all_s"
                x_all_feat.append(x_all_i_j_k_log)
                x_all_s_feat.append(x_all_s_i_j_k)

        # Distill correlation only for the common number of modalities
        student_corr_matrices = compute_correlation_matrix_by_size(x_all_feat)
        teacher_corr_matrices = compute_correlation_matrix_by_size(x_all_s_feat)

        # Compute the MSE loss for each corresponding pair of correlation matrices
        mse_loss = nn.MSELoss()
        total_corr_loss = 0.0
        for student_corr_matrix, teacher_corr_matrix in zip(student_corr_matrices, teacher_corr_matrices):
            if student_corr_matrix.size() != teacher_corr_matrix.size():
                raise ValueError(f"Size mismatch between student and teacher correlation matrices: {student_corr_matrix.size()} vs {teacher_corr_matrix.size()}")
            total_corr_loss += mse_loss(student_corr_matrix, teacher_corr_matrix)
        
    return total_corr_loss / B

def compute_correlation_matrix_by_size(features):
    """
    Compute the pairwise cosine similarity (correlation) between the features of different modalities.
    Handles features of different sizes by grouping tensors by their size.
    
    Args:
        features (list of tensors): List of feature maps for different modalities.
        
    Returns:
        List[torch.Tensor]: List of correlation matrices, one for each group of tensors with the same size.
    """
    from collections import defaultdict
    
    # Group tensors by their shape
    grouped_features = defaultdict(list)
    
    for feature in features:
        shape_key = feature.shape
        grouped_features[shape_key].append(feature)
    
    correlation_matrices = []
    
    # For each group of tensors with the same shape, compute the correlation matrix
    for shape, group in grouped_features.items():
        num_modalities = len(group)
        corr_matrix = torch.zeros((num_modalities, num_modalities), device=group[0].device)
        
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i == j:
                    corr_matrix[i, j] = 1.0  # The correlation of a modality with itself is 1
                else:
                    # Flatten the feature maps and compute cosine similarity between modality i and j
                    f_i = group[i].flatten(1)  # Flatten the feature map except the batch dimension
                    f_j = group[j].flatten(1)
                    cosine_sim = nn.functional.cosine_similarity(f_i, f_j, dim=1).mean()
                    corr_matrix[i, j] = cosine_sim
        
        # Append the correlation matrix for this group of tensors
        correlation_matrices.append(corr_matrix)
    
    return correlation_matrices
