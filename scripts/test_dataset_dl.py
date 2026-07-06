"""Test downloading competitions as Kaggle datasets."""
import kagglehub

comp_dataset_map = {
    'playground-series-s3e18': [
        'kaggle/playground-series-s3e18',
        'spscientist/playground-series-s3e18',
    ],
    'leaf-classification': [
        'kaggle/leaf-classification',
        'c/leaf-classification',
    ],
    'new-york-city-taxi-fare-prediction': [
        'kaggle/new-york-city-taxi-fare-prediction',
        'c/new-york-city-taxi-fare-prediction',
    ],
    'nomad2018-predict-transparent-conductors': [
        'kaggle/nomad2018-predict-transparent-conductors',
        'c/nomad2018-predict-transparent-conductors',
    ],
}

for comp, names in comp_dataset_map.items():
    print(f'\n=== {comp} ===')
    for name in names:
        try:
            path = kagglehub.dataset_download(name, force_download=False)
            print(f'  FOUND as dataset {name}: {path}')
            break
        except Exception as e:
            err = str(e)[:150]
            print(f'  {name}: {err}')
