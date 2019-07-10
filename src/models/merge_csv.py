"""
  Purpose:  Ensemble different .csv files (simple majority vote)
"""

import os

import pandas as pd


def main():
    out_csv_name = "2ch_5best.csv"

    csv_files = [
        "ch_inceptionresnetv2_acc_99.30_loss_0.004159.csv",
        "2ch_densenet169_acc_99.44_loss_0.002444.csv",
        "2ch_inceptionresnetv2_acc_99.68_loss_0.003769.csv",
        "2ch_se_resnext101_32x4d_acc_99.75_loss_0.001964.csv",
        "2ch_densenet169_acc_99.56_loss_0.002460.csv",
    ]

    csv_df = [pd.read_csv(os.path.join("../../models/", x)) for x in csv_files]

    df_out = pd.read_csv(os.path.join("../../models/", csv_files[0]))
    for i in range(1715):
        temp = list()
        for df in csv_df:
            temp.append(df.iloc[i]["class_number"])
        predicted = max(temp, key=temp.count)
        df_out.ix[i, "class_number"] = predicted

    df_out.to_csv(os.path.join("../../models/", out_csv_name), index=False)


if __name__ == "__main__":
    main()
