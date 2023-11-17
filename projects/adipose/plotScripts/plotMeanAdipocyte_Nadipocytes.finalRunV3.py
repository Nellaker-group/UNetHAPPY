import pandas as pd
from matplotlib import pyplot
import statistics as stats
import seaborn as sns


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
    

def plotter(plotlist1,plotlist2,plotname,title="",xlabel='mean size (micrometers**2)'):
    sns.kdeplot(plotlist1,label="vc",color="red", fill=True)
    sns.kdeplot(plotlist2, label="sc",color="blue", fill=True)
    #pyplot.title(title+"\nsize: mean="+str(round(stats.mean(plotlist),2))+", med="+str(round(stats.median(plotlist),2))+", sd="+str(round(stats.stdev(plotlist),2))+", N="+str(len(plotlist)))
    pyplot.xlabel(xlabel)
    pyplot.ylabel('counts')
    pyplot.legend(['vc, N='+str(len(plotlist1)), 'sc, N='+str(len(plotlist2))], title = 'Tissue')
    pyplot.savefig(plotname)
    pyplot.clf()

def plotter2vc(plotlist1,plotlist3,plotlist5,plotlist7,plotlist9,plotname,title="",xlabel='mean size (micrometers**2)'):
    sns.kdeplot(plotlist1,label="leipzig - vc",color="red", fill=True)
    sns.kdeplot(plotlist3,label="munich - vc",color="cyan", fill=True)
    sns.kdeplot(plotlist5,label="hohenheim - vc",color="magenta", fill=True)
    sns.kdeplot(plotlist7,label="endox - vc",color="blue", fill=True)
    sns.kdeplot(plotlist9,label="gtex - vc",color="orange", fill=True)
    #pyplot.title(title+"\nsize: mean="+str(round(stats.mean(plotlist),2))+", med="+str(round(stats.median(plotlist),2))+", sd="+str(round(stats.stdev(plotlist),2))+", N="+str(len(plotlist)))
    pyplot.xlabel(xlabel)
    pyplot.ylabel('counts')
    pyplot.legend(['leipgiz vc, N='+str(len(plotlist1)), 'munich vc, N='+str(len(plotlist3)), 'hohenheim vc, N='+str(len(plotlist5)), 'endox vc, N='+str(len(plotlist7)), 'gtex vc, N='+str(len(plotlist9))], title = 'Tissue')
    pyplot.savefig(plotname)
    pyplot.clf()


def plotter2sc(plotlist2,plotlist4,plotlist6,plotlist8,plotlist10,plotname,title="",xlabel='mean size (micrometers**2)'):
    sns.kdeplot(plotlist2, label="leipzig - sc",color="darkred", fill=True)
    sns.kdeplot(plotlist4, label="munich sc",color="darkcyan", fill=True)
    sns.kdeplot(plotlist6, label="hohenheim - sc",color="darkmagenta", fill=True)
    sns.kdeplot(plotlist8, label="endox - sc",color="darkblue", fill=True)
    sns.kdeplot(plotlist10, label="gtex - sc",color="darkorange", fill=True)
    #pyplot.title(title+"\nsize: mean="+str(round(stats.mean(plotlist),2))+", med="+str(round(stats.median(plotlist),2))+", sd="+str(round(stats.stdev(plotlist),2))+", N="+str(len(plotlist)))
    pyplot.xlabel(xlabel)
    pyplot.ylabel('counts')
    pyplot.legend(['leipgiz sc, N='+str(len(plotlist2)), 'munich sc, N='+str(len(plotlist4)), 'hohenheim sc, N='+str(len(plotlist6)), 'endox sc, N='+str(len(plotlist8)), 'gtex sc, N='+str(len(plotlist10))], title = 'Tissue')
    pyplot.savefig(plotname)
    pyplot.clf()



def plotter3vc(plotlist1,plotlist3,plotlist9,plotname,title="",xlabel='mean size (micrometers**2)'):
    sns.kdeplot(plotlist1,label="leipzig - vc",color="red", fill=True)
    sns.kdeplot(plotlist3,label="munich - vc",color="cyan", fill=True)
    sns.kdeplot(plotlist9,label="gtex - vc",color="orange", fill=True)
    #pyplot.title(title+"\nsize: mean="+str(round(stats.mean(plotlist),2))+", med="+str(round(stats.median(plotlist),2))+", sd="+str(round(stats.stdev(plotlist),2))+", N="+str(len(plotlist)))
    pyplot.xlabel(xlabel)
    pyplot.ylabel('counts')
    pyplot.legend(['leipgiz vc, N='+str(len(plotlist1)), 'munich vc, N='+str(len(plotlist3)), 'gtex vc, N='+str(len(plotlist9))], title = 'Tissue')
    pyplot.savefig(plotname)
    pyplot.clf()


def plotter3sc(plotlist2,plotlist4,plotlist8,plotlist10,plotname,title="",xlabel='mean size (micrometers**2)'):
    sns.kdeplot(plotlist2, label="leipzig - sc",color="darkred", fill=True)
    sns.kdeplot(plotlist4, label="munich sc",color="darkcyan", fill=True)
    sns.kdeplot(plotlist8, label="endox - sc",color="darkblue", fill=True)
    sns.kdeplot(plotlist10, label="gtex - sc",color="darkorange", fill=True)
    #pyplot.title(title+"\nsize: mean="+str(round(stats.mean(plotlist),2))+", med="+str(round(stats.median(plotlist),2))+", sd="+str(round(stats.stdev(plotlist),2))+", N="+str(len(plotlist)))
    pyplot.xlabel(xlabel)
    pyplot.ylabel('counts')
    pyplot.legend(['leipgiz sc, N='+str(len(plotlist2)), 'munich sc, N='+str(len(plotlist4)), 'endox sc, N='+str(len(plotlist8)), 'gtex sc, N='+str(len(plotlist10))], title = 'Tissue')
    pyplot.savefig(plotname)
    pyplot.clf()



df = pd.read_csv('/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/multiRun_Total_from1_to927V2.csv')

## based on 2D histogram of Nadipocytes and avg. size
df=df.loc[ df["Nadipocytes"] > 300,]

endox = pd.read_csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv")

megaListVC = []
megaListSC = []

sub_df_leip_vc = df.loc[ (df["filename"].str.startswith("a")) & (df["filename"].str.contains("vc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_leip_vc.reset_index(drop=True, inplace=True)

sub_df_leip_sc = df.loc[ (df["filename"].str.startswith("a")) & (df["filename"].str.contains("sc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_leip_sc.reset_index(drop=True, inplace=True)

megaListVC.append(extract2(sub_df_leip_vc,what="avg",typeWord="vc")[0])
megaListSC.append(extract2(sub_df_leip_sc,what="avg",typeWord="sc")[0])

    
sub_df_munich_vc = df.loc[ (df["filename"].str.startswith("m")) & (df["filename"].str.contains("vc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_munich_vc.reset_index(drop=True, inplace=True)

sub_df_munich_sc = df.loc[ (df["filename"].str.startswith("m")) & (df["filename"].str.contains("sc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_munich_sc.reset_index(drop=True, inplace=True)

megaListVC.append(extract2(sub_df_munich_vc,what="avg",typeWord="vc")[0])
megaListSC.append(extract2(sub_df_munich_sc,what="avg",typeWord="sc")[0])


sub_df_hohenheim_vc = df.loc[ (df["filename"].str.startswith("h")) & (df["filename"].str.contains("vc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_hohenheim_vc.reset_index(drop=True, inplace=True)

sub_df_hohenheim_sc = df.loc[ (df["filename"].str.startswith("h")) & (df["filename"].str.contains("sc")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_hohenheim_sc.reset_index(drop=True, inplace=True)

megaListVC.append(extract2(sub_df_hohenheim_vc,what="avg",typeWord="vc")[0])
megaListSC.append(extract2(sub_df_hohenheim_sc,what="avg",typeWord="sc")[0])


endox_ids = endox.loc[ endox["depot"].str.startswith("visc")]["filename"].to_list()
sub_df_endox_vc = df.loc[ df["filename"].isin(endox_ids),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_endox_vc.reset_index(drop=True, inplace=True)

endox_ids = endox.loc[ endox["depot"].str.startswith("subcu")]["filename"].to_list()
sub_df_endox_sc = df.loc[ df["filename"].isin(endox_ids),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_endox_sc.reset_index(drop=True, inplace=True)

megaListVC.append(extract2(sub_df_endox_vc,what="avg",typeWord="LLLLLLLLLLLLLL")[0])
megaListSC.append(extract2(sub_df_endox_sc,what="avg",typeWord="LLLLLLLLLLLLLL")[0])

sub_df_gtex_vc = df.loc[ (df["filename"].str.startswith("GTEX")) & (df["filename"].str.contains("Visceral")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_gtex_vc.reset_index(drop=True, inplace=True)

sub_df_gtex_sc = df.loc[ (df["filename"].str.startswith("GTEX")) & (df["filename"].str.contains("Subcutaneous")),("filename","avg","stdev","Nadipocytes","fracBelow750")]
sub_df_gtex_sc.reset_index(drop=True, inplace=True)




megaListVC.append(extract2(sub_df_gtex_vc,what="avg",typeWord="Visceral")[0])
megaListSC.append(extract2(sub_df_gtex_sc,what="avg",typeWord="Subcutaneous")[0])

plotter([item for sublist in megaListVC for item in sublist],[item for sublist in megaListSC for item in sublist],"allCohortsSizeSmoothHistV3.png",title="")


plotter2vc(extract2(sub_df_leip_vc,what="avg",typeWord="vc")[0],extract2(sub_df_munich_vc,what="avg",typeWord="vc")[0],extract2(sub_df_hohenheim_vc,what="avg",typeWord="vc")[0],extract2(sub_df_endox_vc,what="avg",typeWord="LLLLLLLLLLLLLL")[0],extract2(sub_df_gtex_vc,what="avg",typeWord="Visceral")[0],"eachCohortSizeSmoothHistV3vc.png",title="")

plotter2sc(extract2(sub_df_leip_sc,what="avg",typeWord="sc")[0],extract2(sub_df_munich_sc,what="avg",typeWord="sc")[0],extract2(sub_df_hohenheim_sc,what="avg",typeWord="sc")[0],extract2(sub_df_endox_sc,what="avg",typeWord="LLLLLLLLLLLLLL")[0],extract2(sub_df_gtex_vc,what="avg",typeWord="Visceral")[0],"eachCohortSizeSmoothHistV3sc.png",title="")



plotter3vc(extract2(sub_df_leip_vc,what="avg",typeWord="vc")[0],extract2(sub_df_munich_vc,what="avg",typeWord="vc")[0],extract2(sub_df_gtex_vc,what="avg",typeWord="Visceral")[0],"selectCohortsSizeSmoothHistV3vc.png",title="")

plotter3sc(extract2(sub_df_leip_sc,what="avg",typeWord="sc")[0],extract2(sub_df_munich_sc,what="avg",typeWord="sc")[0],extract2(sub_df_endox_sc,what="avg",typeWord="LLLLLLLLLLLLLL")[0],extract2(sub_df_gtex_vc,what="avg",typeWord="Visceral")[0],"selectCohortsSizeSmoothHistV3sc.png",title="")
