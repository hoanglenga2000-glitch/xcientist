import importlib.util
from pathlib import Path
import pandas as pd

spec=importlib.util.spec_from_file_location('siim_preparation',Path(__file__).resolve().parents[1]/'scripts/siim_calibration_prepare_data.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_patient_and_duplicate_groups_remain_in_one_fold():
    frame=pd.DataFrame({'image_name':['im'+str(i) for i in range(100)],'patient_id':['patient'+str(i//2) for i in range(100)],'target':[0,1]*50})
    hashes={image:str(i) for i,image in enumerate(frame.image_name)}
    hashes['im2']=hashes['im0']
    first=module.build_folds(frame,hashes);second=module.build_folds(frame,hashes)
    pd.testing.assert_frame_equal(first,second)
    assert set(first.fold)==set(range(5))
    assert first.loc[0,'group']==first.loc[2,'group']
    joined=frame.join(first[['fold']])
    assert joined.groupby('patient_id').fold.nunique().max()==1
    assert first.groupby('group').fold.nunique().max()==1

def test_missing_patient_ids_do_not_form_one_giant_group():
    frame=pd.DataFrame({'image_name':['im'+str(i) for i in range(100)],'patient_id':[None]*100,'target':[0,1]*50})
    result=module.build_folds(frame,{image:image for image in frame.image_name})
    assert result.group.nunique()==100
