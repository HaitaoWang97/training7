#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import contrib, gluon, image, nd
import numpy as np
np.set_printoptions(2)


# In[6]:


img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[0:2]

print(h, w)
X = nd.random.uniform(shape=(1, 3, h, w))
Y = contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])#确定锚框的大小和高宽比
Y.shape


# In[7]:


#将Y的形状变为（批量大小，锚框个数，4）
boxes = Y.reshape((h, w, 5, 4))
boxes[250, 250, 0, :]  #输出的坐标值已除以图像的高宽，所以在[0,1]之间


# In[11]:


#定义描绘边框的函数
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm','c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                     va='center', ha='center', fontsize=9, color=text_color,
                     bbox=dict(facecolor=color, lw=0))


# In[12]:


d2l.set_figsize()
bbox_scale = nd.array((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
           ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])


# In[14]:


#交并比
#两个真实边框
ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
#五个anchor box
anchors = nd.array([[0, 0.1, 0.2, 0.3],[0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.91]])
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);


# In[15]:


labels = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0),
                                  ground_truth.expand_dims(axis=0),
                                   nd.zeros((1, 3, 5)))


# In[16]:


labels[2]


# In[17]:


labels[1]


# In[18]:


labels[0]


# In[19]:


#输出预测边框
anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                  [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = nd.array([0] * anchors.size)
cls_probs = nd.array([[0] * 4,
                     [0.9, 0.8, 0.7, 0.1],
                     [0.1, 0.2, 0.3, 0.9]])


# In[20]:


fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
           ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])


# In[21]:


output = contrib.ndarray.MultiBoxDetection(
    cls_probs.expand_dims(axis=0), offset_preds.expand_dims(axis=0),
anchors.expand_dims(axis=0), nms_threshold=0.5)
output


# In[ ]:


fig = d2l.plt.imshow(img)
for i in output[0].asnumpy():
    if i[0] == -1:
        continu

