# Kaggle credential readiness

- Release status: `Auth Pending`
- Credential: `Not Configured`
- Real API smoke: `Not Run`

The release source contains no Kaggle credential, user path, CLI path, or cached authentication claim. Install a token through the protected credential manager, then run an explicit real API smoke before claiming readiness.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_kaggle_secret.ps1 install-token
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_kaggle_secret.ps1 smoke -AllowRealExternal
python scripts\verify_kaggle_dpapi_readiness.py --allow-real-external --write-report
```

Official leaderboard submission always requires a Human Gate.
