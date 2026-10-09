import apply_siim_web_bridge as deployment
deployment.STAGE=deployment.BASE/'staging/siim-mlebench-calibration-20260908/pause-recovery'
deployment.BACKUP=deployment.BASE/'backups/siim-pause-recovery-20260910'
deployment.ops.BACKUP=deployment.BACKUP
if __name__=='__main__':deployment.main()
