#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import image


# In[4]:


d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);


# In[5]:


#手动标注边界框
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655,493]


# In[6]:


#定义绘制边界框的函数
def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(
    xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
    fill = False, edgecolor=color, linewidth=2)


# In[7]:


fig =d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));


# In[ ]:




