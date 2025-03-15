import os
import time
def test(model,test_loader):
  os.mkdir(str(round(time.time())))
  for i, (images, targets) in enumerate(test_loader):
    pass