import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import PyPDF2
import warnings


def read_csv(feature_file_path, tpm_file_path):
    eda_feature = pd.read_csv(feature_file_path)
    tpm_with_xgboost = pd.read_csv(tpm_file_path)
    df_need = tpm_with_xgboost[[i[0] for i in eda_feature.values.tolist()]]
    df_need_dataset_Winning = df_need[tpm_with_xgboost['dataset_numeric'] == 1]
    df_need_dataset_Awry = df_need[tpm_with_xgboost['dataset_numeric'] == 0]
    return df_need, df_need_dataset_Winning, df_need_dataset_Awry


def describe_data(df_need_dataset, type=""):
    df_need_dataset_cp = deepcopy(df_need_dataset)
    df_need_dataset_cp.columns = ["{}_{}".format(1 if type == "Winning" else 0, type) + "_" + i for i in
                                  df_need_dataset_cp.columns]
    describe_df = df_need_dataset_cp.describe()
    describe_df = describe_df.T
    describe_df.to_csv('./Dataset {}_{}_describe.csv'.format(1 if type == "Winning" else 0, type), sep=',')


def draw_single_image(winning_data, awry_data, feature_name):
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

    return newwindata, newaydata  # Return resampled data


def draw_images(df_need_dataset_Winning, df_need_dataset_Awry):
    for i, feature in enumerate(["conversation_length"]):  # Iterate over features of interest
        print("{}/{}".format(i + 1, len(df_need_dataset_Winning.columns)))
        newwindata, newaydata = draw_single_image(df_need_dataset_Winning[feature], df_need_dataset_Awry[feature],
                                                  feature)

        # Calculate descriptive statistics on resampled data
        describe_data(pd.DataFrame({'Dataset 1_Winning': newwindata, 'Dataset 0_Awry': newaydata}), type="Resampled")

        # Optionally, save plots to PDF
        if not os.path.exists("./output"):
            os.mkdir("./output")
        with PdfPages('./output/{}.pdf'.format(feature)) as pdf:
            sns.kdeplot(newwindata, color='red', shade=True, label='Dataset 1_Winning', )
            sns.kdeplot(newaydata, color='black', shade=True, label='Dataset 0_Awry', )
            plt.legend()
            pdf.savefig()
            plt.close()


def concat_pdfs(features_name, save_name):
    merger = PyPDF2.PdfMerger()
    for feature_name in features_name:
        pdf_path = "./output/{}.pdf".format(feature_name)
        merger.append(open(pdf_path, "rb"))
    with open(save_name, 'wb') as f:
        merger.write(f)
    merger.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    EDA_features_path = "EDA Features - Sheet1.csv"
    tpm_with_xgboost_noreg_reduced_dim_path = "tpm_with_xgboost_noreg_reduced_dim.csv"

    df_need, df_need_dataset_Winning, df_need_dataset_Awry = read_csv(EDA_features_path,
                                                                      tpm_with_xgboost_noreg_reduced_dim_path)

    # Draw images and compute descriptive statistics after resampling
    draw_images(df_need_dataset_Winning, df_need_dataset_Awry)

    # Optionally, concatenate PDFs if needed
    # concat_pdfs(df_need_dataset_Winning.columns, "./draw_images.pdf")
