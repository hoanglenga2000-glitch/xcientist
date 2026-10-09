"""Common CPU input-loader compatibility mode and optimizer heartbeat; no model choice."""
def configure_serial_dataloader(artifact_root,_trace_handles=[]):
    import inspect,json,time,torch,faulthandler
    original=torch.utils.data.DataLoader.__init__
    original_iter=torch.utils.data.DataLoader.__iter__
    signature=inspect.signature(original)
    def serial_init(self,*args,**kwargs):
        bound=signature.bind(self,*args,**kwargs)
        bound.arguments.update(num_workers=0,pin_memory=False,persistent_workers=False,
                               multiprocessing_context=None,prefetch_factor=None,timeout=0)
        values=dict(bound.arguments);values.pop('self')
        return original(self,**values)
    torch.utils.data.DataLoader.__init__=serial_init
    def threaded_iter(self):
        if self._dataset_kind!=0 or self.batch_sampler is None:
            yield from original_iter(self);return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8,thread_name_prefix='siim-decode') as pool:
            for indices in self.batch_sampler:
                # map retains sampler order; no tensor sharing across subprocesses.
                samples=list(pool.map(self.dataset.__getitem__,indices,timeout=120))
                yield self.collate_fn(samples)
    torch.utils.data.DataLoader.__iter__=threaded_iter
    (artifact_root/'loader-runtime.json').write_text(json.dumps({'num_workers':0,'decode_threads':8,'pin_memory':False,'applies_to_all_arms':True,'model_hyperparameters_unchanged':True}))
    trace=(artifact_root/'runtime-stack.txt').open('w')
    _trace_handles.append(trace)
    faulthandler.dump_traceback_later(180,repeat=True,file=trace)
    count=[0]
    from torch.optim.optimizer import register_optimizer_step_post_hook
    def progress(optimizer,args,kwargs):
        count[0]+=1
        if count[0]==1 or count[0]%10==0:
            temp=artifact_root/'optimizer-progress.json.new'
            temp.write_text(json.dumps({'optimizer_steps':count[0],'at_epoch':time.time(),'gpu_memory_allocated':int(torch.cuda.memory_allocated())}));temp.replace(artifact_root/'optimizer-progress.json')
    register_optimizer_step_post_hook(progress)
    return trace
