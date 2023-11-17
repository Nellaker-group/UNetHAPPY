import pandas as pd
from matplotlib import pyplot
import statistics as stats
import seaborn as sns



df = pd.read_csv('/gpfs3/well/lindgren/users/swf744/adipocyte/data/raw/mergedPhenotypeFile.csv')

endox = pd.read_csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv")



leip_vc = df.loc[ df["ID"].str.startswith("a") &  ~df["ID"].isna(), ("avg_vc")]
leip_vc_list = leip_vc.tolist()

leip_sc = df.loc[ df["ID"].str.startswith("a") &  ~df["ID"].isna(), ("avg_sc")]
leip_sc_list = leip_sc.tolist()


mun_vc = df.loc[ df["ID"].str.startswith("m") &  ~df["ID"].isna(), ("avg_vc")]
mun_vc_list = mun_vc.tolist()

mun_sc = df.loc[ df["ID"].str.startswith("m") &  ~df["ID"].isna(), ("avg_sc")]
mun_sc_list = mun_sc.tolist()


hoh_vc = df.loc[ df["ID"].str.startswith("PAC") &  ~df["ID"].isna(), ("avg_vc")]
hoh_vc_list = hoh_vc.tolist()

hoh_sc = df.loc[ df["ID"].str.startswith("PAC") &  ~df["ID"].isna(), ("avg_sc")]
hoh_sc_list = hoh_sc.tolist()


gtex_vc = df.loc[ df["ID"].str.startswith("GTEX") &  ~df["ID"].isna(), ("avg_vc")]
gtex_vc_list = gtex_vc.tolist()

gtex_sc = df.loc[ df["ID"].str.startswith("GTEX") &  ~df["ID"].isna(), ("avg_sc")]
gtex_sc_list = gtex_sc.tolist()


endox_vc = df.loc[ df["ID"].str.startswith("EX-") &  ~df["ID"].isna(), ("avg_vc")]
endox_vc_list = endox_vc.tolist()

endox_sc = df.loc[ df["ID"].str.startswith("EX-") &  ~df["ID"].isna(), ("avg_sc")]
endox_sc_list = endox_sc.tolist()


###################################


bigList = [leip_sc_list, mun_sc_list, hoh_sc_list, gtex_sc_list, endox_sc_list, leip_vc_list, mun_vc_list, hoh_vc_list, gtex_vc_list, endox_vc_list]

nameList = ["Leipzig subcutaneous ($\mu m^{2}$)", "Munich subcutaneous ($\mu m^{2}$)", "Hohenheim subcutaneous ($\mu m^{2}$)", "GTEX subcutaneous ($\mu m^{2}$)", "ENDOX subcutaneous ($\mu m^{2}$)", "Leipzig visceral ($\mu m^{2}$)", "Munich visceral ($\mu m^{2}$)", "Hohenheim visceral ($\mu m^{2}$)", "GTEX visceral ($\mu m^{2}$)", "ENDOX visceral ($\mu m^{2}$)"]

# Create the figure and axes
fig, axes = pyplot.subplots(2, 5, figsize=(16, 6), sharex=True)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through each subplot

for i, ax in enumerate(axes):

    color = ""
    print(i)
    if i > 4:
        color = "darkgreen"
    else:
        color = "darkblue"
    
    # Plot the histogram
    sns.histplot(bigList[i], ax=ax, kde=False, color=color, alpha=0.4, bins=20, stat="density")
    
    # Plot the smoothed histogram (kernel density estimation)
    sns.kdeplot(bigList[i], ax=ax, color=color, linewidth=2)
    
    # Set titles and labels
    if i > 4:
        ax.set_title(nameList[i].replace("visceral","subcutaneous"),size=12)
    ax.set_ylabel('density')
    ax.set_xlabel(nameList[i],size=12)
  
    


# Adjust spacing between subplots
pyplot.tight_layout()
#pyplot.subplots_adjust(hspace=0.35)

 



pyplot.savefig("plotsPresentationCecilia/Figure3_craigV2.png")
pyplot.clf()


print("DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
