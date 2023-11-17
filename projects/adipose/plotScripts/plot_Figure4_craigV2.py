import pandas as pd
from matplotlib import pyplot
import statistics as stats
import seaborn as sns
import numpy as np
from scipy.stats import linregress


df = pd.read_csv('/gpfs3/well/lindgren/users/swf744/adipocyte/data/raw/mergedPhenotypeFile.csv')

endox = pd.read_csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv")



leip_vc = df.loc[ df["ID"].str.startswith("a") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna(), ("avg_vc")]
leip_vc_list = leip_vc.tolist()

leip_sc = df.loc[ df["ID"].str.startswith("a") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("avg_sc")]
leip_sc_list = leip_sc.tolist()

leip_bmi_vc = df.loc[ df["ID"].str.startswith("a") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna() , ("BMI")]
leip_bmi_vc_list = leip_bmi_vc.tolist()

leip_bmi_sc = df.loc[ df["ID"].str.startswith("a") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna() , ("BMI")]
leip_bmi_sc_list = leip_bmi_sc.tolist()


mun_vc = df.loc[ df["ID"].str.startswith("m") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna(), ("avg_vc")]
mun_vc_list = mun_vc.tolist()

mun_sc = df.loc[ df["ID"].str.startswith("m") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("avg_sc")]
mun_sc_list = mun_sc.tolist()

mun_bmi_vc = df.loc[ df["ID"].str.startswith("m") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna(), ("BMI")]
mun_bmi_vc_list = mun_bmi_vc.tolist()

mun_bmi_sc = df.loc[ df["ID"].str.startswith("m") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("BMI")]
mun_bmi_sc_list = mun_bmi_sc.tolist()


hoh_vc = df.loc[ df["ID"].str.startswith("PAC") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna(), ("avg_vc")]
hoh_vc_list = hoh_vc.tolist()

hoh_sc = df.loc[ df["ID"].str.startswith("PAC") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("avg_sc")]
hoh_sc_list = hoh_sc.tolist()

hoh_bmi_vc = df.loc[ df["ID"].str.startswith("PAC") &  ~df["ID"].isna() & ~df["avg_vc"].isna() &  ~df["BMI"].isna(), ("BMI")]
hoh_bmi_vc_list = hoh_bmi_vc.tolist()

hoh_bmi_sc = df.loc[ df["ID"].str.startswith("PAC") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("BMI")]
hoh_bmi_sc_list = hoh_bmi_sc.tolist()


gtex_vc = df.loc[ df["ID"].str.startswith("GTEX") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna(), ("avg_vc")]
gtex_vc_list = gtex_vc.tolist()

gtex_sc = df.loc[ df["ID"].str.startswith("GTEX") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("avg_sc")]
gtex_sc_list = gtex_sc.tolist()

gtex_bmi_vc = df.loc[ df["ID"].str.startswith("GTEX") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna(), ("BMI")]
gtex_bmi_vc_list = gtex_bmi_vc.tolist()

gtex_bmi_sc = df.loc[ df["ID"].str.startswith("GTEX") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("BMI")]
gtex_bmi_sc_list = gtex_bmi_sc.tolist()



endox_vc = df.loc[ df["ID"].str.startswith("EX-") &  ~df["ID"].isna() & ~df["avg_vc"].isna() & ~df["BMI"].isna(), ("avg_vc")]
endox_vc_list = endox_vc.tolist()

endox_sc = df.loc[ df["ID"].str.startswith("EX-") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("avg_sc")]
endox_sc_list = endox_sc.tolist()

endox_bmi_vc = df.loc[ df["ID"].str.startswith("EX-") &  ~df["ID"].isna() & ~df["avg_vc"].isna() &  ~df["BMI"].isna(), ("BMI")]
endox_bmi_vc_list = endox_bmi_vc.tolist()

endox_bmi_sc = df.loc[ df["ID"].str.startswith("EX-") &  ~df["ID"].isna() & ~df["avg_sc"].isna() & ~df["BMI"].isna(), ("BMI")]
endox_bmi_sc_list = endox_bmi_sc.tolist()



###################################

bmiList = [leip_bmi_sc_list, mun_bmi_sc_list, hoh_bmi_sc_list, gtex_bmi_sc_list, endox_bmi_sc_list, leip_bmi_vc_list, mun_bmi_vc_list, hoh_bmi_vc_list, gtex_bmi_vc_list, endox_bmi_vc_list]
bigList = [leip_sc_list, mun_sc_list, hoh_sc_list, gtex_sc_list, endox_sc_list, leip_vc_list, mun_vc_list, hoh_vc_list, gtex_vc_list, endox_vc_list]

nameList = ["Leipzig subcutaneous ($\mu m^{2}$)", "Munich subcutaneous ($\mu m^{2}$)", "Hohenheim subcutaneous ($\mu m^{2}$)", "GTEX subcutaneous ($\mu m^{2}$)", "ENDOX subcutaneous ($\mu m^{2}$)", "Leipzig visceral ($\mu m^{2}$)", "Munich visceral ($\mu m^{2}$)", "Hohenheim visceral ($\mu m^{2}$)", "GTEX visceral ($\mu m^{2}$)", "ENDOX visceral ($\mu m^{2}$)"]


# Create the figure and axes
fig, axes = pyplot.subplots(2, 5, figsize=(16, 6), sharex=True)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through each subplot
for i, ax in enumerate(axes):


    slope, intercept, r_value, p_value, std_err = linregress(bigList[i],bmiList[i])

    print(slope)
    print(intercept)
    print("")
    
    color = ""
    print(i)
    if i > 4:
        color = "darkgreen"
    else:
        color = "darkblue"
    sns.regplot(x=bigList[i],y=bmiList[i],color=color,scatter_kws={'alpha':0.4}, ax=ax, line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})    
    #ax.set_title(f"y={ax.lines[0].get_slope():.2f}x+{ax.lines[0].get_intercept():.2f}\nR^2={ax.lines[0].rvalue**2:.2f}")
    #ax.set_title(nameList[i],size=10)
    ax.set_xlabel(nameList[i],fontsize=10)
    ax.set_ylabel(ylabel="BMI",fontsize=10)
    ax.xaxis.label.set_size(10)
    ax.annotate("y={0:.4f}x+{1:.2f}\nP={2:.2e}, R2={3:.2f}".format(slope,intercept,p_value,r_value), 
                xy=(1.00, 0.05), # Co-ordinates for positioning
            xycoords='axes fraction',
            ha='right')
    #pyplot.ylim(15,40)


# Adjust spacing between subplots
pyplot.tight_layout()

pyplot.subplots_adjust(hspace=0.25)

pyplot.savefig("plotsPresentationCecilia/Figure4_craigV2.png")
pyplot.clf()


print("DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
