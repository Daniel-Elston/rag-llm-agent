from src.data.data_dict import data_dictionary

def data_dictionary():
    return {
    "dtypes": {
        "age": int,
        "sex": int,
        "cp": int,
        "trtbps": int,
        "output": int
    },
    "use_cols": [
        "age",
        "sex",
        "cp",
        "trtbps",
        "output"
    ],
    "rename_mapping": {
        "age": "a",
        "sex": "s",
        "cp": "c",
        "trtbps": "t"
    },
    "na_values": [
        "?",
        "NA"
    ],
    "label_col": "output",
}