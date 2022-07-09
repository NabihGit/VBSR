import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ep
import matplotlib.pyplot as plt

def MeshBlcok(x, patch_height = 10, patch_width = 10):
    """ MeshBlcok
    """
    n, c, img_h, img_w = x.size()
    assert img_h % patch_height == 0 and img_w % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
    num_h, num_w = img_h // patch_height, img_w // patch_width
    num_patches = num_h * num_w
    #print(x)
    pathes_x = ep.rearrange(x, 'b c (h s1) (w s2) -> b c (h w) (s1 s2)', s1=patch_height, s2=patch_width)
    #print(pathes_x)
    for index_n in range(0, n):
        for index_patch in range(0,num_patches):
            temp = torch.sum(pathes_x[index_n,:,index_patch,:])
            if temp > patch_height*patch_width*0.3 :
                pathes_x[index_n, :, index_patch, :] = 1
            else:
                pathes_x[index_n, :, index_patch, :] = 0
    pathes_x = ep.rearrange(pathes_x, 'b c (h w) (s1 s2) -> b c (h s1) (w s2)', h=num_h, s2=patch_width)
    #print(pathes_x)

    return pathes_x


if __name__ == '__main__':
    x = torch.randn(1,1,12,6).sign()
    MeshBlcok(x)
