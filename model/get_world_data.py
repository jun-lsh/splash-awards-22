import ee
from datetime import timedelta, datetime
from google.cloud import storage

import geopandas as gp
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import os.path

from shapely.ops import unary_union
from shapely.geometry import Polygon, shape, Point

import numpy as np

from datetime import timedelta, datetime
from io import StringIO

from typing import Tuple
from itertools import cycle

import torch.nn as nn
import torch

from torch.utils.data import Subset

from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from time import sleep


# GRU model functions
def conv_output_shape(h_w, kernel_size, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (d,h,w) and returns a tuple of (d,h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


class GRUNet(nn.Module):
    """
    The input is expected to be a 5d tensor of dims [batch_size, time_steps, height, width, n_features]
    representing a height*width region for a certain time_steps period. The region would contain n_features of
    water_velocity, etc. to predict the final output. The final output will be 1 number representing the amount of
    microplastics pieces/m^3.
    """

    def __init__(self, input_dims: Tuple[int], hidden_size: int, output_dim=1, n_layers=1, drop_prob=0):
        super(GRUNet, self).__init__()

        self.batch_size, self.time_steps, self.height, self.width, self.n_features = input_dims

        # hidden size refers to the dimensions of the GRUs hidden state
        self.hidden_size = hidden_size
        # should be 1, as we are predicting only total microplastic concentration in that height*width region
        self.output_dim = output_dim
        # number of stacked GRUs (default 1)
        self.n_layers = n_layers

        # 1*3*3 kernel makes it independent of the time_steps dimension, but n_features being the channel makes sure the
        # convolutions depend on the value of the features of the surrounding pixels
        self.feature_kernel = (1, 3, 3)

        # series of convolutions to reduce dimensionality of height*width*n_features to height*width*1
        # Quite sure ts0 = time_steps as number of time steps should not be changed throughout the convolutions
        # [batch_size, n_features, time_steps, height, width] -> [batch_size, n_features, ts0, h0, w0]
        self.conv0 = nn.Conv3d(self.n_features, self.n_features, self.feature_kernel)
        h0, w0 = conv_output_shape((self.height, self.width), (3, 3))
        # TODO: Add more conv layers
        self.conv1 = nn.Conv3d(self.n_features, self.n_features, self.feature_kernel)
        h1, w1 = conv_output_shape((h0, w0), (3, 3))
        # normalize after convolutions
        self.norm = nn.BatchNorm3d(self.n_features)

        # this final conv layer will learn to compress n_features into 1 number, essentially a 1*1*1*n_features kernel
        # -> [batch_size, 1, ts0, h0, w0]
        self.feature_conv = nn.Conv3d(self.n_features, 1, (1, 1, 1))
        # squeeze out the 1 dim.
        # -> [batch_size, ts0, h0*w0] flattened final feature map of last 2 dims
        self.flatten = nn.Flatten(-2, -1)

        # input size of GRU will be the flattened feature map size = h0 * w0
        self.gru = nn.GRU(h1 * w1, self.hidden_size, self.n_layers, batch_first=True, dropout=drop_prob)
        self.fc0 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc1 = nn.Linear(self.hidden_size // 2, self.output_dim)

        self.relu = nn.ReLU()

    def forward(self, x, h):
        # permute as conv3d accepts inputs of [batch_size, channels/n_features, D/time_steps, H, W]
        x = torch.permute(x, (0, 4, 1, 2, 3))
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.feature_conv(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        # remove the 1 dimensional n_features channel. new shape [batch_size, time_steps, height, width]
        x = torch.squeeze(x)
        # flatten the height x width channel into a final feature map. -> [batch_size, time_steps, height * width]
        x = self.flatten(x)

        out, h = self.gru(x, h)
        out = self.fc0(self.relu(out[:, -1]))
        out = self.fc1(self.relu(out))
        out = torch.squeeze(out)

        # this is not part of the original model, but is added for inference
        # as we do not actually want negative values as output
        out = self.relu(out)

        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.gru.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device)
        return hidden


class CustomImageDataset(Dataset):
    def __init__(self, max_shape=(11, 11), data_file=r"data/latest_atlantic_data.pickle"):
        with open(data_file, 'rb') as handle:
            # train_files holds a list of [label, time_series] pairs
            # where each label themselves are a list of densities for each time frame.
            train_data = pickle.load(handle)

        # the different time series data are all of different shapes. We have to pad them on the fly.
        self.coords, self.images = zip(*train_data)
        self.height, self.width = max_shape

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coords = self.coords[idx]
        image = self.images[idx]

        # calculate margins for paddings
        top = int(np.floor((self.height - image.shape[1]) / 2.0))
        bottom = int(np.ceil((self.height - image.shape[1]) / 2.0))
        left = int(np.floor((self.width - image.shape[2]) / 2.0))
        right = int(np.ceil((self.width - image.shape[2]) / 2.0))

        # pad the image -> [don't pad time, pad height, pad width, don't pad channels]
        image = np.pad(image, [(0, 0), (top, bottom), (left, right), (0, 0)])
        return coords, image


def call_model(*args, **kwargs):
    
    # sacrilege 
    global service_account
    global private_key
    global date
    global timespan
    global sample_n
    global model_path
    global device
    
    # initialize earth engine using json key for service account
    credentials = ee.ServiceAccountCredentials(service_account, private_key)
    ee.Initialize(credentials)

    # read the atlantic ocean shape from shapefile
    print("reading sampling region atlantic")
    tes = gp.read_file(r'data\iho\iho.shp')
    df = tes.loc[tes.name.isin(['South Atlantic Ocean', 'North Atlantic Ocean'])]
    atlantic = df.dissolve().to_crs(epsg=3857)

    # 10 x 10 km wide squares
    length = 10000
    wide = 10000

    # sample random rectangular regions in the atlantic
    def random_rects_in_polygon(number, polygon):
        regions = []
        regions_gpd = gp.GeoDataFrame()
        min_x, min_y, max_x, max_y = polygon.bounds
        while regions_gpd.shape[0] < number:
            for j in range(5000):
                x, y = np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)
                rect = Polygon([(x, y), (x + wide, y), (x + wide, y + length), (x, y + length)])
                regions.append(rect)
            regions_gpd = gp.sjoin(atlantic, gp.GeoDataFrame(regions, geometry=0, crs='epsg:3857'),
                                   how='right').dropna()
        return regions_gpd.iloc[:sample_n]

    # sample regions are stored here
    data_points = random_rects_in_polygon(sample_n, atlantic.geometry[0])

    # this function will query EE api for the data and dump it in a GCS bucket
    def extract_raster_values_from_df(df, image_collection, band_names, task_prefix, date, lookback_days=3, scale=5000,
                                      export=False):
        df['ee_region'] = df.geometry.apply(lambda x: ee.Geometry.Polygon(list((x.exterior.coords)), proj='EPSG:3857'))
        regionCollection = ee.List([])
        emptyCol = ee.FeatureCollection(ee.Feature(None))

        # iterate through all entries
        for row in df.itertuples():
            # instantiate the image collection for the selected time series
            collection = ee.ImageCollection(image_collection).filterDate(
                (date - timedelta(days=lookback_days)).strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')).select(
                band_names)

            # get the raster information based on the buffer region we created
            pixelInfoRegion = collection.getRegion(geometry=getattr(row, 'ee_region'), scale=scale)

            # remove the first element, which is the header ['id', 'longitude', 'latitude', 'time', 'velocity_u_0',
            # 'velocity_v_0']
            pixelInfoRegion = pixelInfoRegion.remove(pixelInfoRegion.get(0))

            # convert the 2d list of information to features. Going to hard code this part as I cba
            def func(x):
                x = ee.List(x)
                feat = ee.Feature(ee.Geometry.Point([x.get(1), x.get(2)], proj='EPSG:4326'),
                                  {'time': x.get(3), 'velocity_u_0': x.get(4), 'velocity_v_0': x.get(5)})
                return feat

            col = ee.FeatureCollection(pixelInfoRegion.map(func))
            # add the region that we created to the overall feature collection

            # ensure collection is not empty bands
            regionCollection = regionCollection.add(ee.Algorithms.If(collection.size(), col, emptyCol))

        if export:
            task_desc = task_prefix.split("/")[1]
            return ee.batch.Export.table.toCloudStorage(collection=ee.FeatureCollection(regionCollection).flatten(),
                                                        bucket='water_velocities', description=task_desc,
                                                        fileNamePrefix=task_prefix, fileFormat="csv")
        else:
            return ee.FeatureCollection(regionCollection).flatten()

    # script keeps crashing when we task.start() for too many points, so we split it into smaller chunks so EE doesn't
    # complain as much
    print("sending api request to earth engine")
    size = 500
    for i in range(0, sample_n, size):
        task_prefix = f"{date.strftime('%d.%m.%Y')}/atlantic_water_velocities_{i}"
        task = extract_raster_values_from_df(data_points.iloc[i:i + size].copy(), 'HYCOM/sea_water_velocity',
                                             ['velocity_u_0', 'velocity_v_0'], task_prefix, date,
                                             lookback_days=timespan, export=True)
        task.start()

    print("waiting 10 minutes (average) for task completion")
    sleep(600)

    # initialise GCS credentials
    client = storage.Client.from_service_account_json(json_credentials_path=private_key)
    bucket = client.get_bucket('water_velocities')

    # query GCS for the existence of the last datafile, after which we know all files are done, and we can continue
    blob = bucket.blob(f'{date.strftime("%d.%m.%Y")}/atlantic_water_velocities_{sample_n - size}.csv')
    while not blob.exists():
        # wait 60 seconds before checking again
        print("csv files unavailable, waiting one minute")
        sleep(60)
        print("checking for csv files on GCS")

    print("csv files available, processing now")
    # at this point, all the csv data files are in the GCS bucket, and we are going to combine them into a dataframe
    df_list = []
    for i in range(0, sample_n, size):
        # retrieve the blob from the GCS bucket
        blob = bucket.blob(f'{date.strftime("%d.%m.%Y")}/atlantic_water_velocities_{i}.csv')
        blob = blob.download_as_string().decode('utf-8')
        # convert into stringIO object for pandas to read it as a csv file
        blob = StringIO(blob)

        sub_df = pd.read_csv(blob)

        water_idx = sub_df['system:index'].str.split('_', expand=True)
        water_idx = water_idx.rename(columns={0: 'cluster', 1: 'index'})
        water_idx['cluster'] = water_idx['cluster'].astype(int) + i

        sub_df = sub_df.join(water_idx).drop(columns=['system:index'])

        df_list.append(sub_df)

    water_df = pd.concat(df_list, axis=0)

    # remove clusters with NA values
    water_df['.geo'] = water_df['.geo'].apply(json.loads).apply(shape)
    water_df = water_df.dropna().set_index(['cluster'])
    remap = dict(zip(water_df.index.unique(), list(range(len(water_df.index.unique())))))
    water_df = water_df.rename(index=remap).set_index(['index'], append=True)
    water_df = gp.GeoDataFrame(water_df, geometry='.geo')

    # process the dataframe into format our model can read and dump into a pickle file
    images = []
    largestx, largesty = -1, -1
    counter = 0
    for key, cluster_df in water_df.groupby(level=0):
        cluster_df = cluster_df.droplevel(0)

        cluster_df.time = pd.to_datetime(cluster_df.time, unit='ms').dt.strftime('%Y-%m-%d')
        cluster_df.time = pd.Categorical(cluster_df.time)

        cluster_df.time = cluster_df.time.cat.rename_categories(list(range(len(cluster_df.time.cat.categories))))

        cluster_df['x'] = pd.Categorical(cluster_df.geometry.x)
        cluster_df['y'] = pd.Categorical(cluster_df.geometry.y)
        cluster_df.x = cluster_df.x.cat.rename_categories(list(range(len(cluster_df.x.cat.categories))))
        cluster_df.y = cluster_df.y.cat.rename_categories(list(range(len(cluster_df.y.cat.categories))))

        time, maxx, maxy = cluster_df.time.cat.categories[-1] + 1, cluster_df.x.cat.categories[-1] + 1, \
                           cluster_df.y.cat.categories[-1] + 1

        # i don't want to deal with this shit
        if time != timespan:
            counter += 1
            continue

        cluster_df = cluster_df.pivot_table(values=['velocity_u_0', 'velocity_v_0'], index=['time', 'y', 'x'],
                                            aggfunc=np.sum)
        img = cluster_df.to_numpy().reshape([time, maxy, maxx, -1])
        largestx, largesty = max(maxx, largestx), max(maxy, largesty)

        # [(x, y), img]
        pos = water_df.loc[key].iloc[0].loc['.geo']
        images.append([(pos.x, pos.y), img])

    print("processing complete, dumping into pickle file")
    with open('data/latest_atlantic_data.pickle', 'wb') as handle:
        pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # instantiate the GRU model for inference
    # copied the same parameters as the model was trained with.
    hidden_dim = 128
    batch_size = 128
    time_steps = 7
    height = 11
    width = 11
    n_features = 2
    output_dim = 1
    n_layers = 1
    drop_prob = 0
    input_dim = (batch_size, time_steps, height, width, n_features)

    # Run model on the data. We might end up with less data because drop_last=True is set to prevent model from
    # breaking when the total dataset size is not divisible by batch_size
    print("instantiating GRU model for inference")
    # load model in eval mode
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, drop_prob)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    dataset = CustomImageDataset(max_shape=(11, 11), data_file='data/latest_atlantic_data.pickle')
    loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=True)

    # list of batch_size tensors
    print("running inference")
    results = list()
    with torch.inference_mode():
        for coords, img in loader:
            h = model.init_hidden(batch_size)
            output, h = model.forward(img.to(device).float(), h)
            results.append(torch.stack(coords + [output.cpu()], dim=1))

    results = torch.cat(results)

    # write model predictions to output file
    print("writing inference results to output file model_preds.json")
    f = open("model_preds.json", "w")
    f.write(json.dumps([{'lat': lat, 'lng': lng, 'conc': conc} for lat, lng, conc in results.tolist()]))
    f.close()


def initCreds():

    # sacrilege 
    global service_account
    global private_key
    global date
    global timespan
    global sample_n
    global model_path
    global device

    service_account = '860927269044-compute@developer.gserviceaccount.com'
    private_key = 'keys/splash-awards-telegram-bot-b9f732990190.json'

    # HYCOM dataset usually has a 2 day delay for satellite data
    date = datetime.today() - timedelta(days=2)
    # time frame for the model
    timespan = 7
    # number of points to sample from the atlantic, and request api data for to predict
    sample_n = 10000
    # filepath of model to use for inference
    model_path = r"model_checkpoints/GRU_epoch_1500_loss_3.401.pt"

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

if __name__ == "__main__":
    initCreds()
    call_model()
