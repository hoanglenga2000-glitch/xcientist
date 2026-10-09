"""Check unprivileged isolation availability without changing mounts or system settings."""
import ctypes,json,subprocess

def main():
    libc=ctypes.CDLL(None,use_errno=True)
    abi=libc.syscall(444,ctypes.c_void_p(0),ctypes.c_size_t(0),ctypes.c_uint(1))
    result={'landlock_abi':abi,'landlock_errno':ctypes.get_errno() if abi<0 else 0}
    p=subprocess.run(['unshare','--user','--map-root-user','true'],capture_output=True,timeout=10)
    result.update(unprivileged_user_namespace_available=p.returncode==0,unshare_exit_code=p.returncode,
                  persistent_mounts_created=False,system_configuration_changed=False,training_started=False)
    print(json.dumps(result))

if __name__=='__main__':main()
