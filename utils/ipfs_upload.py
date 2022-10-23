from utils import get_keys
from pinatapy import PinataPy
from model.model import GRUNet
import torch
import pandas as pd

pinata = None


def initPinata():
    global pinata
    pinata_keys = get_keys("./../api_keys.json", ["PINATA_API_KEY", "PINATA_API_SECRET"])
    pinata = PinataPy(pinata_keys[0], pinata_keys[1])


def pinFile(path):
    result = pinata.pin_file_to_ipfs(path)
    hash = result["IpfsHash"]
    return hash


def init_model_eval(model_path):
    # copied the same parameters as the model was trained with.
    hidden_dim = 64
    batch_size = 128
    time_steps = 3
    height = 11
    width = 11
    n_features = 2
    output_dim = 1
    n_layers = 1
    drop_prob = 0
    input_dim = (batch_size, time_steps, height, width, n_features)

    # load model in eval mode
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, drop_prob)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def water_velocities(date):
    # go to https://developers.google.com/earth-engine/datasets/catalog/HYCOM_sea_water_velocity and check the most
    # recent update date. then this one will retrieve the 3 consecutive dates of time series images (date param is last)
    # and process them for model input
    return pd.DataFrame()


# remember to run this script from the root directory
if __name__ == "__main__":
    # automated model inference
    #model_path = r"model/gru_model.pt"
    #model = init_model_eval(model_path)

    initPinata()
    hash = pinFile("base_heatmap.json")

    file = open("hashes.txt", "a")
    file.write(f"{hash}\n")
    file.close()
