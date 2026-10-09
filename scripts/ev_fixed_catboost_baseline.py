"""The predeclared fixed baseline, not an EvoMind-generated candidate."""
import numpy as np
from catboost import CatBoostClassifier


class Predictor:
    def __init__(self,model,categories):
        self.model=model
        self.categories=categories
        self.classes_=np.array([0,1])

    def predict_proba(self,x):
        prepared=x.copy()
        for name in self.categories:
            prepared[name]=prepared[name].fillna('__MISSING__').astype(str)
        return self.model.predict_proba(prepared)


def fit_model(x_train,y_train,x_valid,y_valid,seed):
    categories=x_train.select_dtypes(exclude='number').columns.tolist()
    training=x_train.copy()
    for name in categories:
        training[name]=training[name].fillna('__MISSING__').astype(str)
    model=CatBoostClassifier(iterations=1000,depth=6,learning_rate=0.05,loss_function='Logloss',
                             eval_metric='AUC',random_seed=seed,task_type='GPU',devices='0',
                             thread_count=8,allow_writing_files=False,verbose=False)
    if x_valid is not None:
        validation=x_valid.copy()
        for name in categories:
            validation[name]=validation[name].fillna('__MISSING__').astype(str)
        model.fit(training,y_train,cat_features=categories,eval_set=(validation,y_valid),early_stopping_rounds=100,verbose=False)
    else:
        model.fit(training,y_train,cat_features=categories,verbose=False)
    return Predictor(model,categories)
