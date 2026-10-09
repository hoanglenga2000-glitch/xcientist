"""Toy/stub contract checks only: no competition data, model training, GPU or network."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from research_agent_workstation import tabular_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location('ev_test_'+name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_script('validate_tabular_experiment')
onboard = load_script('onboard_kaggle_competition')


class StubClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, forbid_predict=True):
        self.forbid_predict = forbid_predict

    def fit(self, x, y):
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, x):
        p = np.asarray(x)[:, 0].astype(float)
        return np.column_stack((p, 1-p))

    def predict(self, x):
        if self.forbid_predict:
            raise AssertionError('AUC path must not call predict')
        return (np.asarray(x)[:, 0] > .5).astype(int)


def config(metric='roc_auc'):
    return {'task':{'type':'classification','metric':metric,'id_column':'id','target':'target','prediction_column':'target'},
            'thresholds':{},'scaffold':{'split_seed':20260908}}


def test_positive_class_order_is_explicit():
    stub = StubClassifier().fit(None, None)
    assert pipeline.positive_class_probabilities(stub, pd.DataFrame({'p':[.2,.8]})).tolist() == [.2,.8]


@pytest.mark.parametrize('values', [[np.nan],[np.inf],[-.01],[1.01]])
def test_invalid_probabilities_fail(values):
    with pytest.raises(ValueError, match='finite'):
        pipeline.positive_class_probabilities(StubClassifier().fit(None,None), pd.DataFrame({'p':values}))


def test_missing_positive_label_and_multiclass_fail():
    stub=StubClassifier().fit(None,None)
    for classes in [np.array([2,3]), np.array([0,1,2])]:
        stub.classes_=classes
        with pytest.raises(ValueError,match='two classes'):
            pipeline.positive_class_probabilities(stub,pd.DataFrame({'p':[.4]}))


@pytest.mark.parametrize('metric',['roc_auc','auc','roc_auc_score'])
def test_auc_path_scores_oof_probabilities_without_label_predictions(monkeypatch,metric):
    monkeypatch.setattr(pipeline,'selected_models',lambda *a:{'stub':StubClassifier()})
    monkeypatch.setattr(pipeline,'build_preprocessor',lambda x:'passthrough')
    x=pd.DataFrame({'p':[.1,.9]*20});y=pd.Series([0,1]*20)
    result,_=pipeline.evaluate_classification(x,y,config(metric),17)
    assert result['selection_metric']=='oof_roc_auc'
    assert result['split_seed']==20260908
    metrics=result['model_results']['stub']
    assert metrics['oof_roc_auc']==1 and len(metrics['fold_roc_auc'])==5
    assert 'cv_accuracy_mean' not in metrics
    assert pipeline.metrics_pass(config(metric),result)
    assert validator.validate_metric(config(metric),{'evaluation':result})['oof_roc_auc']==1


def test_accuracy_behavior_unchanged(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,'selected_models',lambda *a:{'stub':StubClassifier(False)})
    monkeypatch.setattr(pipeline,'build_preprocessor',lambda x:'passthrough')
    x=pd.DataFrame({'p':[.1,.9]*20});y=pd.Series([0,1]*20)
    result,model=pipeline.evaluate_classification(x,y,config('accuracy'),17)
    assert result['model_results']['stub']['cv_accuracy_mean']==1
    sample=pd.DataFrame({'id':[4,5],'target':[0,0]})
    checks=pipeline.make_submission(model,x.iloc[:2],sample,config('accuracy'),tmp_path)
    assert checks['valid'] and pd.read_csv(tmp_path/'submission.csv')['target'].tolist()==[0,1]


def test_auc_submission_keeps_probabilities_and_ids(tmp_path):
    sample=pd.DataFrame({'id':[4,5],'target':[0,0]})
    checks=pipeline.make_submission(StubClassifier().fit(None,None),pd.DataFrame({'p':[.2,.8]}),sample,config(),tmp_path,test_ids=sample.id)
    assert checks['valid'] and checks['ids_match']
    assert pd.read_csv(tmp_path/'submission.csv').target.tolist()==[.2,.8]


@pytest.mark.parametrize('ids',[[5,4],[4,4],[4,np.nan]])
def test_auc_export_rejects_id_errors_before_writing(tmp_path,ids):
    sample=pd.DataFrame({'id':[4,5],'target':[0,0]})
    with pytest.raises(ValueError,match='IDs'):
        pipeline.make_submission(StubClassifier().fit(None,None),pd.DataFrame({'p':[.2,.8]}),sample,config(),tmp_path,test_ids=pd.Series(ids))
    assert not (tmp_path/'submission.csv').exists()


@pytest.mark.parametrize('bad', ['order','duplicate','nan','infinity','range'])
def test_disk_submission_validator_checks_probabilities_and_ids(tmp_path,bad):
    sample=pd.DataFrame({'id':[4,5],'target':[0,0]})
    sample.to_csv(tmp_path/'sample.csv',index=False)
    pd.DataFrame({'id':[4,5],'p':[.2,.8]}).to_csv(tmp_path/'test.csv',index=False)
    submission=pd.DataFrame({'id':[4,5],'target':[.2,.8]})
    if bad=='order':submission['id']=[5,4]
    elif bad=='duplicate':submission['id']=[4,4]
    else:submission.loc[0,'target']={'nan':np.nan,'infinity':np.inf,'range':1.1}[bad]
    submission.to_csv(tmp_path/'submission.csv',index=False)
    cfg=config();cfg['data']={'test':str(tmp_path/'test.csv'),'sample_submission':str(tmp_path/'sample.csv')}
    log={'submission_check':{'path':str(tmp_path/'submission.csv'),'valid':True,'missing_predictions':0},
         'data_quality':{'train_test_feature_columns_match':True}}
    with pytest.raises(SystemExit,match='VALIDATION_FAILED'):
        validator.validate_submission(cfg,log,tmp_path)


@pytest.mark.parametrize('metric',['accuracy','roc_auc'])
def test_onboarding_retains_accuracy_labels_but_allows_auc_probabilities(tmp_path,monkeypatch,metric):
    monkeypatch.setattr(onboard,'ROOT',tmp_path)
    train=pd.DataFrame({'id':[1,2],'p':[.1,.9],'target':[0,1]})
    test=pd.DataFrame({'id':[3],'p':[.2]});sample=pd.DataFrame({'id':[3],'target':[0]})
    cfg=onboard.build_config('test','test',train,test,sample,'target','classification',metric,
                            {'train':'train.csv','test':'test.csv','sample_submission':'sample.csv'},tmp_path/'task')
    assert ('allowed_prediction_values' in cfg['thresholds']) == (metric=='accuracy')
    assert cfg['thresholds'].get('require_probability_predictions',False) == (metric=='roc_auc')


def test_auc_validation_does_not_accept_accuracy_scores():
    log={'evaluation':{'metric':'accuracy','best_model':'stub','model_results':{'stub':{'cv_accuracy_mean':1}}}}
    with pytest.raises(SystemExit,match='different evaluation'):
        validator.validate_metric(config(),log)


def test_onboarding_official_yes_no_positive_label(tmp_path,monkeypatch):
    monkeypatch.setattr(onboard,'ROOT',tmp_path)
    train=pd.DataFrame({'id':[1,2],'p':[.1,.9],'target':['No','Yes']})
    test=pd.DataFrame({'id':[3],'p':[.2]});sample=pd.DataFrame({'id':[3],'target':[0.]})
    cfg=onboard.build_config('test','test',train,test,sample,'target','classification','roc_auc',
                            {'train':'train.csv','test':'test.csv','sample_submission':'sample.csv'},tmp_path/'task')
    assert cfg['task']['positive_label']=='Yes'
    assert 'allowed_prediction_values' not in cfg['thresholds']
