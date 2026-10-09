"""Activate completion-recovery using the existing backed-up deployment contract."""
import apply_siim_web_bridge as deployment

deployment.STAGE=deployment.BASE/'staging/siim-mlebench-calibration-20260908/completion-recovery'
deployment.BACKUP=deployment.BASE/'backups/siim-completion-recovery-20260909'
deployment.ops.BACKUP=deployment.BACKUP

if __name__=='__main__':
    deployment.main()
