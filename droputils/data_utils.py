import configparser
import os
from datetime import datetime, date, time
import numpy as np

import droputils.rough_segments as segments


def get_config(config_file="../orcestra_drop.cfg"):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def get_l3_path(config, flight_id="20240811", platform="HALO"):
    """
    get l3 filename for a flight from config file
    """
    l3_file = os.path.join(
        config["processor.Gridded.get_l3_dir"]["l3_dir"],
        config["processor.Gridded.get_l3_filename"]["l3_filename_template"].format(
            platform=platform
        ),
    )
    return l3_file


def get_circle_data(ds, flight_id="20240811"):
    """
    get a dictionary of circle data for one flight
    """
    print("HELELELELELE")

    flight_date = date.fromisoformat(flight_id)
    circles = {
        circle: {
            "start_time": np.datetime64(
                datetime.combine(
                    flight_date, time.fromisoformat(segments.starts[flight_id][circle])
                )
            ),
            "end_time": np.datetime64(
                datetime.combine(
                    flight_date, time.fromisoformat(segments.ends[flight_id][circle])
                )
            ),
        }
        for circle in segments.starts[flight_id].keys()
    }
    ds_c = {}
    for circle in list(circles.keys()):
        try:
            ds_c[circle] = ds.where(
                ds["launch_time"].astype("datetime64")
                > circles[circle]["start_time"],
                drop=True,
            ).where(
                ds["launch_time"].astype("datetime64")
                < circles[circle]["end_time"],
                drop=True,
            )
        except ValueError:
            print(f"No sondes for circle {circle}. It is omitted")

    return ds_c
