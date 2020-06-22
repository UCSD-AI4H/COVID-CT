#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().system('git clone https://github.com/Adamdad/moco.git')


# In[4]:


get_ipython().system("python main_moco.py   -a densenet169   --lr 0.015   --batch-size 128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed 4 --world-size 1 --rank 0 --resume save_model_dense/checkpoint_luna_covid_moco_backup.pth.tar")
# --resume save_model_dense/checkpoint_covid.pth.tar


# In[ ]:




