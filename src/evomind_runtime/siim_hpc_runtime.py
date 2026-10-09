"""SIIM uses its verified isolated interpreter, without a second dependency install."""
import json,shlex
from .hpc_runtime_overlay import HpcRuntime

class SiimHpcRuntime(HpcRuntime):
    verified_python=''

    def prepare_solution_environment(self,script_path):
        prefix='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/siim_job90353/runtime/'
        python=self.verified_python
        if not python.startswith(prefix) or not python.endswith('/venv/bin/python'):raise ValueError('siim_runtime_path_rejected')
        program=('import json,torch,torchvision,numpy,pandas,sklearn,sys;'
                 'v={"torch":torch.__version__,"torchvision":torchvision.__version__,"numpy":numpy.__version__,"pandas":pandas.__version__,"sklearn":sklearn.__version__};'
                 'assert v=={"torch":"2.5.1+cu118","torchvision":"0.20.1+cu118","numpy":"1.26.4","pandas":"2.2.3","sklearn":"1.7.2"};'
                 'print(json.dumps({"status":"verified_existing_runtime","versions":v,"network_installations":0}))')
        client=self._connector()
        try:
            limit=self.remaining_seconds(90,margin=60)
            rc,out,err=self._exec(client,shlex.quote(python)+' -I -c '+shlex.quote(program),timeout=limit)
            if rc:raise ValueError('siim_existing_runtime_import_failed')
            result=json.loads(out)
            if result.get('status')!='verified_existing_runtime':raise ValueError('siim_runtime_receipt_invalid')
            if self.progress_callback:self.progress_callback(source='managed_adapter',work_kind='preparing',phase='existing_siim_runtime_verified',worker_state='observed',detail='Verified existing SIIM Python environment; no dependency download')
            return result
        finally:client.close()
