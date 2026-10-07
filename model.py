import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

NAMES = ['checking','duration','history','purpose','amount','savings','employment','installment_rate','personal_status','guarantor','residence','property','age','other_plans','housing','existing_credits','job','dependents','telephone','foreign_worker']
NUMERIC = ['duration','amount','installment_rate','existing_credits']
CATEGORIES = ['checking','history','purpose','savings','employment','guarantor','property','other_plans','housing']
FIELDS = ['checking','duration','history','purpose','amount','savings','employment','installment_rate','guarantor','property','other_plans','housing','existing_credits']
LABELS = {'checking':'Checking account category','duration':'Loan duration (months)','history':'Credit history category','purpose':'Loan purpose category','amount':'Loan amount (historical DM)','savings':'Savings category','employment':'Employment duration category','installment_rate':'Installment rate (dataset level 1-4)','guarantor':'Other debtors / guarantors','property':'Property category','other_plans':'Other installment plans','housing':'Housing category','existing_credits':'Existing credits at this bank'}

class CreditModel:
    def __init__(self):
        self.frame = pd.read_csv(Path(__file__).parent/'data'/'german.data', sep=r'\s+', names=NAMES+['target'])
        self.target = (self.frame.target == 2).astype(int)
        self.train_ids,self.test_ids = train_test_split(np.arange(len(self.frame)),test_size=.2,stratify=self.target,random_state=42)
        numeric = NUMERIC + ['log_amount','monthly_principal']
        preprocessing = ColumnTransformer([('numbers',StandardScaler(),numeric),('categories',OneHotEncoder(handle_unknown='ignore'),CATEGORIES)])
        self.pipeline = make_pipeline(preprocessing,LogisticRegression(max_iter=2000,random_state=42))
        features = self.engineer(self.frame[FIELDS])
        self.pipeline.fit(features.iloc[self.train_ids],self.target.iloc[self.train_ids])
        y=self.target.iloc[self.test_ids]
        pred=self.pipeline.predict(features.iloc[self.test_ids])
        probability=self.pipeline.predict_proba(features.iloc[self.test_ids])[:,1]
        matrix=confusion_matrix(y,pred,labels=[0,1])
        self.metrics=dict(train_samples=len(self.train_ids),test_samples=len(self.test_ids),accuracy=float(accuracy_score(y,pred)),precision_bad=float(precision_score(y,pred,zero_division=0)),recall_bad=float(recall_score(y,pred,zero_division=0)),f1_bad=float(f1_score(y,pred,zero_division=0)),roc_auc_bad=float(roc_auc_score(y,probability)),confusion_matrix=matrix.tolist(),confusion_labels=['good','bad'],threshold=.5,majority_baseline_accuracy=float((y==0).mean()),uci_weighted_error_cost=int(matrix[0,1]+5*matrix[1,0]))
        self.report=classification_report(y,pred,target_names=['good','bad'],digits=4)
        self.options={name:sorted(self.frame.iloc[self.train_ids][name].unique()) for name in CATEGORIES}

    @staticmethod
    def engineer(frame):
        result=frame.copy()
        result['log_amount']=np.log1p(result['amount'].astype(float))
        result['monthly_principal']=result['amount'].astype(float)/result['duration'].astype(float)
        return result

    def predict(self,values):
        parsed={}
        for name in FIELDS:
            if name in NUMERIC:
                try: value=float(values[name])
                except (ValueError,KeyError): raise ValueError('Enter a number for '+LABELS[name]) from None
                if not np.isfinite(value) or value<=0: raise ValueError(LABELS[name]+' must be a positive finite number.')
                if name!='amount' and value!=int(value): raise ValueError(LABELS[name]+' must be a whole number.')
                if name=='installment_rate' and value not in [1,2,3,4]: raise ValueError('Installment rate must be a dataset level from 1 to 4.')
                parsed[name]=value
            else:
                value=values.get(name,'')
                if value not in self.options[name]: raise ValueError('Choose a valid category for '+LABELS[name])
                parsed[name]=value
        x=self.engineer(pd.DataFrame([parsed]))
        score=float(self.pipeline.predict_proba(x)[0,1])
        return ('bad' if score>=.5 else 'good'),score

    def save_reports(self):
        folder=Path(__file__).parent/'results';folder.mkdir(exist_ok=True)
        cv=cross_val_score(self.pipeline,self.engineer(self.frame[FIELDS]).iloc[self.train_ids],self.target.iloc[self.train_ids],cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='roc_auc')
        metrics=dict(self.metrics,training_cv_auc_mean=float(cv.mean()),training_cv_auc_std=float(cv.std()),model='Logistic Regression',positive_class='bad credit risk',seed=42)
        (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
        (folder/'evaluation.txt').write_text(self.report+'\nConfusion matrix: '+str(metrics['confusion_matrix'])+'\nRows actual, columns predicted; good, bad.\nTraining-only 5-fold ROC-AUC: '+str(cv.mean())+'\nUCI weighted error cost: '+str(metrics['uci_weighted_error_cost'])+'\n')
        return metrics

if __name__=='__main__': print(json.dumps(CreditModel().save_reports(),indent=2))
