"""Download s3e18 via kagglehub dataset."""
import kagglehub, os, shutil

# Try dataset download (playground series data is often available as dataset)
datasets = [
    'kaggle/playground-series-s3e18',
    'spscientist/playground-series-s3e18',
]

for ds in datasets:
    try:
        path = kagglehub.dataset_download(ds, force_download=False)
        print(f'SUCCESS as dataset {ds}: {path}')
        # Copy to prepared dir
        dest = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_prepared/playground-series-s3e18/'
        os.makedirs(dest, exist_ok=True)
        for f in os.listdir(path):
            src = os.path.join(path, f)
            dst = os.path.join(dest, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f'  Copied: {f}')
        print('Done!')
        break
    except Exception as e:
        err = str(e)
        if '403' in err or 'Forbidden' in err:
            print(f'{ds}: 403 FORBIDDEN')
        else:
            print(f'{ds}: {err[:200]}')
else:
    # Last resort: try competition download
    print('Dataset download failed, trying competition download...')
    os.system('/hpc2hdd/home/aimslab/.local/bin/kaggle competitions download -c playground-series-s3e18 -p /tmp/kaggle_dl_s3e18 2>&1')
    print('Exit:', os.path.exists('/tmp/kaggle_dl_s3e18/playground-series-s3e18.zip'))
