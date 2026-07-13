"""Test which competitions are accessible."""
import kagglehub

# Test competitions that we KNOW work (spaceship data is already on server)
test_comps = [
    # Existing (should work)
    'spaceship-titanic',
    'tabular-playground-series-dec-2021',
    'tabular-playground-series-may-2022',
    # New (may not work)
    'playground-series-s3e18',
    'leaf-classification',
    'new-york-city-taxi-fare-prediction',
    'nomad2018-predict-transparent-conductors',
]

for comp in test_comps:
    try:
        path = kagglehub.competition_download(comp)
        print(f'{comp}: OK -> {path}')
    except Exception as e:
        err = str(e)
        if '403' in err or 'Forbidden' in err:
            print(f'{comp}: 403 FORBIDDEN (rules not accepted)')
        else:
            print(f'{comp}: {err[:150]}')
