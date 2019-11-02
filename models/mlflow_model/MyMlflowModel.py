from mlflow import pyfunc
import os
import pandas as pd

class MyMlflowModel(object):

    def __init__(self):
        #next(os.walk('mlruns\\0'))[1][0]
        #self.pyfunc_model = pyfunc.load_pyfunc("mlruns\\0\\"+next(os.walk('mlruns\\0'))[1][0]+"\\artifacts\\model")
        artifact_dir = ""
        print(os.listdir("."))
        for (root,dirs,files) in os.walk('mlruns/0', topdown=True):
            artifact_dir = dirs[0]
            break
        self.pyfunc_model = pyfunc.load_model("s3://tiger-mlflow/1/afa14f38393e4ca5a1a600dbfb43ae73/artifacts/model/")
        #self.pyfunc_model = pyfunc.load_model("mlruns/0/49ff4f88066649dc8f04d136a50770f8/artifacts/model")
        
    def predict(self,X,features_names):
        if not features_names is None and len(features_names)>0:
            df = pd.DataFrame(data=X,columns=features_names)
        else:
            df = pd.DataFrame(data=X)
        return self.pyfunc_model.predict(df)


