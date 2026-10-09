"""Use the established backed-up activation; only response metadata changes."""
import apply_siim_web_bridge as deployment
deployment.STAGE=deployment.BASE/'staging/siim-mlebench-calibration-20260908/response-diagnostics'
deployment.BACKUP=deployment.BASE/'backups/siim-response-diagnostics-20260909'
deployment.ops.BACKUP=deployment.BACKUP
if __name__=='__main__':deployment.main()
