"""Backed-up, idle-only deployment of SIIM request limits."""
import apply_siim_web_bridge as deployment
deployment.STAGE=deployment.BASE/'staging/siim-mlebench-calibration-20260908/web-request-limits'
deployment.BACKUP=deployment.BASE/'backups/siim-web-request-limits-20260910'
deployment.ops.BACKUP=deployment.BACKUP
if __name__=='__main__':deployment.main()
