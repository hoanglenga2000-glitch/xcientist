"""Emergency entry: serve new work, never automatically replay historical Runs."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.runtime import AgentRuntime


def serve_manual(workspace, port):
    runtime=AgentRuntime(Path(workspace).resolve())
    server=ThreadingHTTPServer(('127.0.0.1',port),make_handler(runtime,ensure_token(runtime.runtime_root)))
    try:
        # Deliberately no recover_incomplete: current/retired historical work
        # remains intact and requires explicit reviewed continuation.
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--port',required=True,type=int)
    args=parser.parse_args()
    serve_manual(args.workspace,args.port)
