

import pandas as pd
from matplotlib import pyplot
import statistics as stats
import seaborn as sns



def extract2(sub_df,what="avg",typeWord="vc"):
    ids_dict = {}
    avg_top = {}
    avg_bottom = {}
    avg = []
    ids_list = []
    for i in range(len(sub_df)):
        id=sub_df.loc[i,"filename"].split(typeWord)[0]
        if  sub_df.loc[i,"fracBelow750"] > 0.9:
            continue
        if id in ids_dict:
            avg_top[id] += sub_df.loc[i,"avg"]*sub_df.loc[i,"Nadipocytes"]
            avg_bottom[id] += sub_df.loc[i,"Nadipocytes"]
        else:
            avg_top[id] = sub_df.loc[i,"avg"]*sub_df.loc[i,"Nadipocytes"]
            avg_bottom[id] = sub_df.loc[i,"Nadipocytes"]
            ids_list.append(id)
        ids_dict[id] = 1
    for id in ids_list:
        avg.append(avg_top[id]/avg_bottom[id])
    return((avg,ids_list))



def plotter3vc(plotlist1,plotname,title="",xlabel='mean size (micrometers**2)'):
    sns.kdeplot(plotlist1,label="leipzig - vc",color="red", fill=True)
    pyplot.xlabel(xlabel)
    pyplot.ylabel('counts')
    pyplot.savefig(plotname)
    pyplot.clf()


df = pd.read_csv('/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/multiRun_Total_from1_to927V2.csv')

## based on 2D histogram of Nadipocytes and avg. size
df=df.loc[ df["Nadipocytes"] > 300,]

endox = pd.read_csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv")


sub_df_leip_vc = df.loc[ (df["filename"].str.startswith("a")) & (df["filename"].str.contains("vc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_leip_vc.reset_index(drop=True, inplace=True)

sub_df_leip_sc = df.loc[ (df["filename"].str.startswith("a")) & (df["filename"].str.contains("sc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_leip_sc.reset_index(drop=True, inplace=True)

plotter3vc(extract2(sub_df_leip_vc,what="avg",typeWord="vc")[0],"Figure1_craig.png",title="")


