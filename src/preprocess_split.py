import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

if __name__ == '__main__':

    annotations_path = Path("./data/annotations/")
    data = []
    for instance in os.listdir(annotations_path):
        tree = ET.parse(annotations_path/instance)
        root = tree.getroot()
        attributes = {"file": root[1].text,
                      "width": root[2][0].text,
                      "height": root[2][1].text,
                      "depth": root[2][2].text,
                      "class": root[4][0].text,
                      "xmin": root[4][5][0].text,
                      "ymin": root[4][5][1].text,
                      "xmax": root[4][5][2].text,
                      "ymax": root[4][5][3].text
                      }
        data.append(attributes)

    df = pd.DataFrame.from_records(data)
    test = df.groupby("class").apply(lambda x: x.sample(frac=0.2, random_state=23))
    train = df[~df.file.isin(test.file)]

    df.to_csv("data/all_data.csv")
    train.to_csv("data/train.csv")
    test.reset_index(drop=True).to_csv("data/test.csv")
