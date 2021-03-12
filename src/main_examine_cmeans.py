import pathlib
import os
from datetime import datetime
from tqdm import tqdm

from loadingData import loadSktime
from transformingData import derivatives, normalizing, cmeans
from examiningData import displaying

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

if __name__ == "__main__":

    os.mkdir(plots_dir)

    test_path=pathlib.Path('./data/Univariate/ACSF1/ACSF1_TEST.ts')
    train_path=pathlib.Path('./data/Univariate/ACSF1/ACSF1_TRAIN.ts')
    derivative_order=1
    test_xses_series, test_ys = loadSktime.load_sktime(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = loadSktime.load_sktime(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    input_size = 2
    no_centers = 3
    m = 2.0

    centerss = []
    for _ in tqdm(range(20)):
        fcm_centers, train_xses_series_transformed = cmeans.find_centers_and_transform(
            xses_series=train_xses_series,
            c=no_centers,
            m=m)
        centerss.append(fcm_centers)
    
    displaying.display_series_with_different_markers(train_xses_series,
                                                     plots_dir / 'centers.png',
                                                     "consecutive cmeans runs with default initialization",
                                                     ["" for _ in centerss],
                                                     centerss)
    
    print(centerss)
