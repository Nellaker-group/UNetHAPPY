import pandas as pd
from matplotlib import pyplot
import statistics as stats

def extract(sub_df,what="avg",typeWord="vc"):
    ids_dict = {}
    avg = []
    ids_list = []
    for i in range(len(sub_df)):
        id=sub_df.loc[i,"filename"].split(typeWord)[0]
        if id in ids_dict:
            continue
        else:
            ids_list.append(id)
            avg.append(sub_df.loc[i,what])
        ids_dict[id] = 1        
    return((avg,ids_list))

def extract2(sub_df,what="avg",typeWord="vc"):
    ids_dict = {}
    avg_top = {}
    avg_bottom = {}
    avg = []
    ids_list = []
    
    #emil
    print("sub_df.head()")
    print(sub_df.head())
    
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
    

def plotter(plotlist,plotname,title="",xlabel='mean size (micrometers**2)'):
    pyplot.hist(plotlist, 20, alpha=0.5, color="blue",edgecolor='black', linewidth=1.2)
    pyplot.title(title+"\nsize: mean="+str(round(stats.mean(plotlist),2))+", med="+str(round(stats.median(plotlist),2))+", sd="+str(round(stats.stdev(plotlist),2))+", N="+str(len(plotlist)))
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

plotter(extract2(sub_df_leip_vc,what="avg",typeWord="vc")[0],"plots/leipzig_meanArea_vcV2.png",title="leipzig vc")
plotter(extract2(sub_df_leip_sc,what="avg",typeWord="sc")[0],"plots/leipzig_meanArea_scV2.png",title="leipzig sc")    

    
sub_df_munich_vc = df.loc[ (df["filename"].str.startswith("m")) & (df["filename"].str.contains("vc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_munich_vc.reset_index(drop=True, inplace=True)

sub_df_munich_sc = df.loc[ (df["filename"].str.startswith("m")) & (df["filename"].str.contains("sc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_munich_sc.reset_index(drop=True, inplace=True)

plotter(extract2(sub_df_munich_vc,what="avg",typeWord="vc")[0],"plots/munich_meanArea_vcV2.png",title="munich vc")
plotter(extract2(sub_df_munich_sc,what="avg",typeWord="sc")[0],"plots/munich_meanArea_scV2.png",title="munich sc")    


sub_df_hohenheim_vc = df.loc[ (df["filename"].str.startswith("h")) & (df["filename"].str.contains("vc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_hohenheim_vc.reset_index(drop=True, inplace=True)

sub_df_hohenheim_sc = df.loc[ (df["filename"].str.startswith("h")) & (df["filename"].str.contains("sc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_hohenheim_sc.reset_index(drop=True, inplace=True)

plotter(extract2(sub_df_hohenheim_vc,what="avg",typeWord="vc")[0],"plots/hohenheim_meanArea_vcV2.png",title="hohenheim vc")
plotter(extract2(sub_df_hohenheim_sc,what="avg",typeWord="sc")[0],"plots/hohenheim_meanArea_scV2.png",title="hohenheim sc")    

endox_ids = endox.loc[ endox["depot"].str.startswith("visc")]["filename"].to_list()
sub_df_endox_vc = df.loc[ df["filename"].isin(endox_ids),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_endox_vc.reset_index(drop=True, inplace=True)

endox_ids = endox.loc[ endox["depot"].str.startswith("subcu")]["filename"].to_list()
sub_df_endox_sc = df.loc[ df["filename"].isin(endox_ids),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_endox_sc.reset_index(drop=True, inplace=True)

plotter(extract2(sub_df_endox_vc,what="avg",typeWord="LLLLLLLLLLLLLL")[0],"plots/endox_meanArea_vcV2.png",title="endox vc")
plotter(extract2(sub_df_endox_sc,what="avg",typeWord="LLLLLLLLLLLLLL")[0],"plots/endox_meanArea_scV2.png",title="endox sc")    

sub_df_gtex_vc = df.loc[ (df["filename"].str.startswith("GTEX")) & (df["filename"].str.contains("Visceral")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_gtex_vc.reset_index(drop=True, inplace=True)

sub_df_gtex_sc = df.loc[ (df["filename"].str.startswith("GTEX")) & (df["filename"].str.contains("Subcutaneous")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_gtex_sc.reset_index(drop=True, inplace=True)

plotter(extract2(sub_df_gtex_vc,what="avg",typeWord="Visceral")[0],"plots/GTEX_meanArea_vcV2.png",title="GTEX vc")
plotter(extract2(sub_df_gtex_sc,what="avg",typeWord="Subcutaneous")[0],"plots/GTEX_meanArea_scV2.png",title="GTEX sc")    


#########################################
#########################################

sub_df_leip_vc = df.loc[ (df["filename"].str.startswith("a")) & (df["filename"].str.contains("vc")),("filename","Nadipocytes","fracBelow750")]
sub_df_leip_vc.reset_index(drop=True, inplace=True)

sub_df_leip_sc = df.loc[ (df["filename"].str.startswith("a")) & (df["filename"].str.contains("sc")),("filename","Nadipocytes","fracBelow750")]
sub_df_leip_sc.reset_index(drop=True, inplace=True)

plotter(extract(sub_df_leip_vc,what="Nadipocytes",typeWord="vc")[0],"plots/leipzig_Nadipocytes_vcV2.png",title="leipzig vc",xlabel="Adipocytes per individual")
plotter(extract(sub_df_leip_sc,what="Nadipocytes",typeWord="sc")[0],"plots/leipzig_Nadipocytes_scV2.png",title="leipzig sc",xlabel="Adipocytes per individual")


sub_df_munich_vc = df.loc[ (df["filename"].str.startswith("m")) & (df["filename"].str.contains("vc")),("filename","Nadipocytes","fracBelow750")]
sub_df_munich_vc.reset_index(drop=True, inplace=True)

sub_df_munich_sc = df.loc[ (df["filename"].str.startswith("m")) & (df["filename"].str.contains("sc")),("filename","Nadipocytes","fracBelow750")]
sub_df_munich_sc.reset_index(drop=True, inplace=True)

plotter(extract(sub_df_munich_vc,what="Nadipocytes",typeWord="vc")[0],"plots/munich_Nadipocytes_vcV2.png",title="munich vc",xlabel="Adipocytes per individual")
plotter(extract(sub_df_munich_sc,what="Nadipocytes",typeWord="sc")[0],"plots/munich_Nadipocytes_scV2.png",title="munich sc",xlabel="Adipocytes per individual")


sub_df_hohenheim_vc = df.loc[ (df["filename"].str.startswith("h")) & (df["filename"].str.contains("vc")),("filename","Nadipocytes","fracBelow750")]
sub_df_hohenheim_vc.reset_index(drop=True, inplace=True)

sub_df_hohenheim_sc = df.loc[ (df["filename"].str.startswith("h")) & (df["filename"].str.contains("sc")),("filename","Nadipocytes","fracBelow750")]
sub_df_hohenheim_sc.reset_index(drop=True, inplace=True)

plotter(extract(sub_df_hohenheim_vc,what="Nadipocytes",typeWord="vc")[0],"plots/hohenheim_Nadipocytes_vcV2.png",title="hohenheim vc",xlabel="Adipocytes per individual")
plotter(extract(sub_df_hohenheim_sc,what="Nadipocytes",typeWord="sc")[0],"plots/hohenheim_Nadipocytes_scV2.png",title="hohenheim sc",xlabel="Adipocytes per individual")


endox_ids = endox.loc[ endox["depot"].str.startswith("visc")]["filename"].to_list()
sub_df_endox_vc = df.loc[ df["filename"].isin(endox_ids),("filename","Nadipocytes")]
sub_df_endox_vc.reset_index(drop=True, inplace=True)

endox_ids = endox.loc[ endox["depot"].str.startswith("subcu")]["filename"].to_list()
sub_df_endox_sc = df.loc[ df["filename"].isin(endox_ids),("filename","Nadipocytes")]
sub_df_endox_sc.reset_index(drop=True, inplace=True)

plotter(extract(sub_df_endox_vc,what="Nadipocytes",typeWord="LLLLLLLLLLLLLL")[0],"plots/endox_Nadipocytes_vcV2.png",title="endox vc",xlabel="Adipocytes per individual")
plotter(extract(sub_df_endox_sc,what="Nadipocytes",typeWord="LLLLLLLLLLLLLL")[0],"plots/endox_Nadipocytes_scV2.png",title="endox sc",xlabel="Adipocytes per individual")


sub_df_gtex_vc = df.loc[ (df["filename"].str.startswith("GTEX")) & (df["filename"].str.contains("Visceral")),("filename","Nadipocytes","fracBelow750")]
sub_df_gtex_vc.reset_index(drop=True, inplace=True)

sub_df_gtex_sc = df.loc[ (df["filename"].str.startswith("GTEX")) & (df["filename"].str.contains("Subcutaneous")),("filename","Nadipocytes","fracBelow750")]
sub_df_gtex_sc.reset_index(drop=True, inplace=True)

plotter(extract(sub_df_gtex_vc,what="Nadipocytes",typeWord="Visceral")[0],"plots/GTEX_Nadipocytes_vcV2.png",title="GTEX vc",xlabel="Adipocytes per individual")
plotter(extract(sub_df_gtex_sc,what="Nadipocytes",typeWord="Subcutaneous")[0],"plots/GTEX_Nadipocytes_scV2.png",title="GTEX sc",xlabel="Adipocytes per individual")
