import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import warnings

def read_csv(tpm_file_path):
    tpm_with_xgboost = pd.read_csv(tpm_file_path)
    df_need_dataset_Winning = tpm_with_xgboost[tpm_with_xgboost['dataset_numeric'] == 1]
    df_need_dataset_Awry = tpm_with_xgboost[tpm_with_xgboost['dataset_numeric'] == 0]
    return df_need_dataset_Winning, df_need_dataset_Awry

def resample_data(winning_data, awry_data):
    winningcount = pd.value_counts(winning_data)
    awrycount = pd.value_counts(awry_data)
    indexall = winningcount.index.tolist() + awrycount.index.tolist()
    newcountdict = {}
    for length in indexall:
        if length in winningcount.index.tolist():
            wincount = winningcount[length]
        else:
            wincount = 0
        if length in awrycount.index.tolist():
            aycount = awrycount[length]
        else:
            aycount = 0
        newcountdict[length] = wincount if wincount < aycount else aycount

    newwindata = pd.Series([0], index=[0])
    newaydata = pd.Series([0], index=[0])

    for key, value in newcountdict.items():
        newwindata = pd.concat([newwindata, winning_data[winning_data == key].sample(n=value)])
        newaydata = pd.concat([newaydata, awry_data[awry_data == key].sample(n=value)])
    if not os.path.exists("./output"):
        os.mkdir("./output")
    newwindata.to_csv('./output/Dataset_conversation_length_resampled.csv', index=False)
    return newwindata, newaydata  # Return resampled data

def describe_data(df_need_dataset, type=""):
    df_need_dataset_cp = deepcopy(df_need_dataset)
    df_need_dataset_cp.columns = ["{}_{}".format(1 if type == "Winning" else 0, type) + "_" + i for i in
                                  df_need_dataset_cp.columns]
    describe_df = df_need_dataset_cp.describe()
    describe_df = describe_df.T
    describe_df.to_csv('./output/Dataset {}_{}_describe.csv'.format(1 if type == "Winning" else 0, type), sep=',')

def draw_image(newwindata, newaydata, feature="conversation_length"):
    describe_data(pd.DataFrame({'Dataset 1_Winning': newwindata, 'Dataset 0_Awry': newaydata}), type="Resampled")
    # Optionally, save plots to PDF
    with PdfPages('./output/{}.pdf'.format(feature)) as pdf:
        sns.kdeplot(newwindata, color='red', shade=True, label='Dataset 1_Winning', )
        sns.kdeplot(newaydata, color='black', shade=True, label='Dataset 0_Awry', )
        plt.legend()
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    tpm_with_xgboost_noreg_reduced_dim_path = "tpm_with_xgboost_noreg_reduced_dim.csv"
    df_need_dataset_Winning, df_need_dataset_Awry = read_csv(tpm_with_xgboost_noreg_reduced_dim_path)
    feature = "conversation_length"
    newwindata, newaydata = resample_data(df_need_dataset_Winning[feature], df_need_dataset_Awry[feature])
    draw_image(newwindata, newaydata, feature=feature)
