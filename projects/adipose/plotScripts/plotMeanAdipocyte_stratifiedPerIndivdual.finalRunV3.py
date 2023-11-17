import pandas as pd
from matplotlib import pyplot
import statistics as stats
import seaborn as sns


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

df = pd.read_csv('/gpfs3/well/lindgren/users/swf744/adipocyte/data/raw/mergedPhenotypeFile.csv')

endox = pd.read_csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv")



leip_vc_male = df.loc[ (df["ID"].str.startswith("a")) & (df["Sex"] == 1.0) ,("avg_vc")]
leip_vc_male.reset_index(drop=True, inplace=True)
leip_vc_male_list = leip_vc_male.tolist()

leip_vc_female = df.loc[ (df["ID"].str.startswith("a")) & (df["Sex"] == 2.0) ,("avg_vc")]
leip_vc_female.reset_index(drop=True, inplace=True)
leip_vc_female_list = leip_vc_female.tolist()


sns.kdeplot(leip_vc_male_list, label="leipzig - vc - male",color="darkred", fill=True)
sns.kdeplot(leip_vc_female_list, label="leipzig vc - female",color="darkcyan", fill=True)
pyplot.xlabel("mean adipocyte size")
pyplot.ylabel('counts')
pyplot.legend(['leipgiz vc male, N='+str(len(leip_vc_male_list)), 'leipgiz vc female, N='+str(len(leip_vc_female_list))], title = 'Tissue')
pyplot.savefig("histSizeLeipzigbySex_VC.png")
pyplot.clf()


leip_vc_obese = df.loc[ (df["ID"].str.startswith("a")) & (df["BMI"] > 30.0) ,("avg_vc")]
leip_vc_obese.reset_index(drop=True, inplace=True)
leip_vc_obese_list = leip_vc_obese.tolist()

leip_vc_nonobese = df.loc[ (df["ID"].str.startswith("a")) & (df["BMI"] <= 30.0) ,("avg_vc")]
leip_vc_nonobese.reset_index(drop=True, inplace=True)
leip_vc_nonobese_list = leip_vc_nonobese.tolist()


sns.kdeplot(leip_vc_obese_list, label="leipzig - vc - obese",color="darkred", fill=True)
sns.kdeplot(leip_vc_nonobese_list, label="leipzig vc - nonobese",color="darkcyan", fill=True)
pyplot.xlabel("mean adipocyte size")
pyplot.ylabel('counts')
pyplot.legend(['leipgiz vc obese, N='+str(len(leip_vc_obese_list)), 'leipgiz vc nonobese, N='+str(len(leip_vc_nonobese_list))], title = 'Tissue')
pyplot.savefig("histSizeLeipzigifObese_VC.png")
pyplot.clf()

###################################



leip_sc_male = df.loc[ (df["ID"].str.startswith("a")) & (df["Sex"] == 1.0) ,("avg_sc")]
leip_sc_male.reset_index(drop=True, inplace=True)
leip_sc_male_list = leip_sc_male.tolist()

leip_sc_female = df.loc[ (df["ID"].str.startswith("a")) & (df["Sex"] == 2.0) ,("avg_sc")]
leip_sc_female.reset_index(drop=True, inplace=True)
leip_sc_female_list = leip_sc_female.tolist()


sns.kdeplot(leip_sc_male_list, label="leipzig - sc - male",color="darkred", fill=True)
sns.kdeplot(leip_sc_female_list, label="leipzig sc - female",color="darkcyan", fill=True)
pyplot.xlabel("mean adipocyte size")
pyplot.ylabel('counts')
pyplot.legend(['leipgiz sc male, N='+str(len(leip_sc_male_list)), 'leipgiz sc female, N='+str(len(leip_sc_female_list))], title = 'Tissue')
pyplot.savefig("histSizeLeipzigbySex_SC.png")
pyplot.clf()


leip_sc_obese = df.loc[ (df["ID"].str.startswith("a")) & (df["BMI"] > 30.0) ,("avg_sc")]
leip_sc_obese.reset_index(drop=True, inplace=True)
leip_sc_obese_list = leip_sc_obese.tolist()

leip_sc_nonobese = df.loc[ (df["ID"].str.startswith("a")) & (df["BMI"] <= 30.0) ,("avg_sc")]
leip_sc_nonobese.reset_index(drop=True, inplace=True)
leip_sc_nonobese_list = leip_sc_nonobese.tolist()


sns.kdeplot(leip_sc_obese_list, label="leipzig - sc - obese",color="darkred", fill=True)
sns.kdeplot(leip_sc_nonobese_list, label="leipzig sc - nonobese",color="darkcyan", fill=True)
pyplot.xlabel("mean adipocyte size")
pyplot.ylabel('counts')
pyplot.legend(['leipgiz sc obese, N='+str(len(leip_sc_obese_list)), 'leipgiz sc nonobese, N='+str(len(leip_sc_nonobese_list))], title = 'Tissue')
pyplot.savefig("histSizeLeipzigifObese_SC.png")
pyplot.clf()


############################################

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
