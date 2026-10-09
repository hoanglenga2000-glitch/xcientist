import json,sys
from types import SimpleNamespace
from evomind_runtime.siim_dataloader_runtime import configure_serial_dataloader

def test_common_loader_limits_and_optimizer_progress(tmp_path,monkeypatch):
    hooks=[]
    class Loader:
        def __init__(self,dataset,num_workers=0,pin_memory=True,persistent_workers=False,multiprocessing_context=None,prefetch_factor=None,timeout=0):
            self.num_workers=num_workers;self.pin_memory=pin_memory;self.timeout=timeout
            self.dataset=dataset;self._dataset_kind=0;self.batch_sampler=[[0,1],[2,3]];self.collate_fn=lambda x:x
        def __iter__(self):return iter(self.dataset)
    torch=SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(DataLoader=Loader)),cuda=SimpleNamespace(memory_allocated=lambda:123))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'torch.optim.optimizer',SimpleNamespace(register_optimizer_step_post_hook=lambda fn:hooks.append(fn)))
    monkeypatch.setattr('faulthandler.dump_traceback_later',lambda *a,**k:None)
    trace=configure_serial_dataloader(tmp_path)
    try:
        loader=Loader([0,1,2,3],num_workers=8,pin_memory=True)
        assert loader.num_workers==0 and loader.pin_memory is False and loader.timeout==0
        assert list(loader)==[[0,1],[2,3]]
        for i in range(10):hooks[0](None,None,None)
        assert json.loads((tmp_path/'optimizer-progress.json').read_text())['optimizer_steps']==10
    finally:trace.close()
