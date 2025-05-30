"""
This is the class for setting the parameters for dataset and ML training
"""

class Config:
      
   dataset_root = '../data'
   dataset_root2 = 'new_pred'
   # To use multiple compounds' data
   dataset = ['Sr8La24Co7Ni7Mn7Fe7Cr4O96', 'Ba4Sr4La24Co7Ni7Mn7Fe7Cr4O96', 'Ba2Sr6La24Co7Ni7Mn7Fe7Cr4O96',  'Sr8La24Co7Ni7Mn7Fe7Zn4O96', 'Sr32Co7Ni7Mn7Fe7Cr4O96', 'Sr32Co8Ni8Mn8Fe8O96', 'Ba4Sr4La24Co8Ni8Mn8Fe8O96', 'La32Co8Ni8Mn8Fe8O96', 'Sr8La24Co8Ni8Mn8Fe8O96']

                  
config = Config

