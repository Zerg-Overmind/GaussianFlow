# GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation
[ArXiv](https://arxiv.org/pdf/2403.12365.pdf) | [Project Page](https://zerg-overmind.github.io/GaussianFlow.github.io/)

Please refer to [this repo](https://github.com/Zerg-Overmind/diff-gaussian-rasterization) for our cuda implementations of variables for calculating Gaussian flow.

For now, we refer the code below to calculate Gaussian flow with variables from above. 

```python
### We detach the variables related to t_1 in calculation of GaussianFlow such that the gradient backward 
### only works for variables at t_2 while keeping variables at t_1 unchanged because
### variables at t_1 have been updated at t_1 - 1 with the same logic. 

### This can accelerate the training process since less variables needed to be updated. BTW, not detach 
#### variables at t_1 will not decrase the performance but slow down the training.

# Gaussian parameters at t_1
proj_2D_t_1 = render_t_1["proj_2D"]
gs_per_pixel = render_t_1["gs_per_pixel"].long() 
weight_per_gs_pixel = render_t_1["weight_per_gs_pixel"]
x_mu = render_t_1["x_mu"]
cov2D_inv_t_1 = render_t_1["conic_2D"].detach()

# Gaussian parameters at t_2
proj_2D_t_2 = render_t_2["proj_2D"]
cov2D_inv_t_2 = render_t_2["conic_2D"]
cov2D_t_2 = render_t_2["conic_2D_inv"]


cov2D_t_2_mtx = torch.zeros([cov2D_t_2.shape[0], 2, 2]).cuda()
cov2D_t_2_mtx[:, 0, 0] = cov2D_t_2[:, 0]
cov2D_t_2_mtx[:, 0, 1] = cov2D_t_2[:, 1]
cov2D_t_2_mtx[:, 1, 0] = cov2D_t_2[:, 1]
cov2D_t_2_mtx[:, 1, 1] = cov2D_t_2[:, 2]

cov2D_inv_t_1_mtx = torch.zeros([cov2D_inv_t_1.shape[0], 2, 2]).cuda()
cov2D_inv_t_1_mtx[:, 0, 0] = cov2D_inv_t_1[:, 0]
cov2D_inv_t_1_mtx[:, 0, 1] = cov2D_inv_t_1[:, 1]
cov2D_inv_t_1_mtx[:, 1, 0] = cov2D_inv_t_1[:, 1]
cov2D_inv_t_1_mtx[:, 1, 1] = cov2D_inv_t_1[:, 2]

# B_t_2
U_t_2 = torch.svd(cov2D_t_2_mtx)[0]
S_t_2 = torch.svd(cov2D_t_2_mtx)[1]
V_t_2 = torch.svd(cov2D_t_2_mtx)[2]
B_t_2 = torch.bmm(torch.bmm(U_t_2, torch.diag_embed(S_t_2)**(1/2)), V_t_2.transpose(1,2))

# B_t_1 ^(-1)
U_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[0]
S_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[1]
V_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[2]
B_inv_t_1 = torch.bmm(torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1)**(1/2)), V_inv_t_1.transpose(1,2))

# calculate B_t_2*B_inv_t_1
B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)

# calculate cov2D_t_2*cov2D_inv_t_1
# cov2D_t_2cov2D_inv_t_1 = torch.zeros([cov2D_inv_t_2.shape[0],2,2]).cuda()
# cov2D_t_2cov2D_inv_t_1[:, 0, 0] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1]
# cov2D_t_2cov2D_inv_t_1[:, 0, 1] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 2]
# cov2D_t_2cov2D_inv_t_1[:, 1, 0] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 1]
# cov2D_t_2cov2D_inv_t_1[:, 1, 1] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 2]

# isotropic version of GaussianFlow
#predicted_flow_by_gs = (proj_2D_next[gs_per_pixel] - proj_2D[gs_per_pixel].detach()) * weights.detach()

# full formulation of GaussianFlow
cov_multi = (B_t_2_B_inv_t_1[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze()
predicted_flow_by_gs = (cov_multi + proj_2D_next[gs_per_pixel] - proj_2D[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) * weights.detach()

# flow supervision loss 
large_motion_msk = torch.norm(optical_flow, p=2, dim=-1) >= flow_thresh  # flow_thresh = 0.1 or other value to filter out noise, here we assume that we have already loaded pre-computed optical flow somewhere as pseudo GT
Lflow = torch.norm((optical_flow - predicted_flow_by_gs.sum(0))[large_motion_msk], p=2, dim=-1).mean() 
loss = loss + flow_weight * Lflow # flow_weight could be 1, 0.1, ... whatever you want.

```
