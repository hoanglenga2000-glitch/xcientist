"""Linux-only, fail-closed filesystem and network isolation for SIIM workers."""
from __future__ import annotations
import ctypes,errno,os,stat

READ=(1<<0)|(1<<2)|(1<<3)
WRITE=(1<<16)-1

def restrict_filesystem(read_paths,write_paths,device_paths):
    class RuleSet(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
    libc=ctypes.CDLL(None,use_errno=True)
    abi=libc.syscall(444,ctypes.c_void_p(0),ctypes.c_size_t(0),ctypes.c_uint(1))
    if abi<5:raise RuntimeError('landlock_abi_5_required')
    attr=RuleSet(WRITE)
    ruleset=libc.syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
    if ruleset<0:raise RuntimeError('landlock_ruleset_failed')
    try:
        for paths,mask in [(read_paths,READ),(write_paths,WRITE),(device_paths,(1<<2)|(1<<1)|(1<<15))]:
            for raw in paths:
                path=os.path.realpath(raw)
                if not os.path.exists(path):raise RuntimeError('landlock_allowlist_path_missing')
                fd=os.open(path,os.O_PATH|os.O_CLOEXEC)
                try:
                    directory=stat.S_ISDIR(os.fstat(fd).st_mode)
                    # Directory-only rights cannot be attached to regular files.
                    allowed=mask if directory else mask & ((1<<0)|(1<<1)|(1<<2)|(1<<14)|(1<<15))
                    rule=PathRule(allowed,fd)
                    if libc.syscall(445,ruleset,1,ctypes.byref(rule),0)<0:raise RuntimeError('landlock_path_rule_failed')
                finally:os.close(fd)
        if libc.prctl(38,1,0,0,0)!=0:raise RuntimeError('no_new_privileges_failed')
        if libc.syscall(446,ruleset,0)<0:raise RuntimeError('landlock_restriction_failed')
    finally:os.close(ruleset)
    return {'landlock_abi':abi,'filesystem_restricted':True,'no_new_privileges':True}

def block_internet():
    class Comparison(ctypes.Structure):
        _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_int),('a',ctypes.c_uint64),('b',ctypes.c_uint64)]
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Comparison)]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    context=lib.seccomp_init(0x7fff0000)
    if not context:raise RuntimeError('network_filter_creation_failed')
    try:
        number=lib.seccomp_syscall_resolve_name(b'socket')
        for family in (2,10):
            comparison=Comparison(0,4,family,0)
            if lib.seccomp_rule_add_array(context,0x00050000|errno.EPERM,number,1,ctypes.byref(comparison))!=0:
                raise RuntimeError('network_filter_rule_failed')
        if lib.seccomp_load(context)!=0:raise RuntimeError('network_filter_activation_failed')
    finally:lib.seccomp_release(context)
    import socket
    for family in (socket.AF_INET,socket.AF_INET6):
        try:attempt=socket.socket(family,socket.SOCK_STREAM)
        except PermissionError:continue
        else:attempt.close();raise RuntimeError('network_filter_probe_failed')
    return {'internet_socket_creation_blocked':True}
