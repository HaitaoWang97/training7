#!/usr/bin/env python
# coding: utf-8

# In[9]:


#由于基于原图每个像素生成锚框，数量级过大，计算量太大
#将原图输入卷积层后得到特征图，基于特征图生成锚框，在不同的特征图中生成的锚框对应原图中不同尺寸的感受野
get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import contrib, image, nd

img = image.imread('../img/catdog.jpg')
h, w = img.shape[0: 2]
h, w


# In[17]:


#定义display_anchors函数
d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h))
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)


# In[18]:


display_anchors(fmap_w=4, fmap_h=4, s=[0.15])


# In[20]:


display_anchors(fmap_w=2, fmap_h=2, s=[0.4])


# In[21]:


display_anchors(fmap_w=1, fmap_h=1, s=[0.8])


# In[ ]:




