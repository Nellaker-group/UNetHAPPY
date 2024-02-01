from matplotlib import pyplot
import seaborn as sns
import numpy as np

from projects.adipose.analysis.get_pixel_size import get_which_pixel

def plot_one_hist(dict0, plot_name, plot_title):
    colours = {}
    colours["leipzig"]="green"
    colours["munich"]="red"
    colours["hohenheim"]="blue"
    colours["gtex"]="brown"
    colours["endox"]="black"
    for key in dict0.keys():
        which_pixel = get_which_pixel(key)
        sns.histplot(data=dict0[key], kde=True, element="poly", color=colours[which_pixel], stat="probability",fill=False)
    pyplot.plot([], [], label='leipzig', color='green',alpha=0.5)
    pyplot.plot([], [], label='munich', color='red')
    pyplot.plot([], [], label='hohenheim', color='blue')
    pyplot.plot([], [], label='gtex', color='brown')
    pyplot.plot([], [], label='endox', color='black')
    pyplot.xlim(0, 15000)
    pyplot.xlabel('Size of adipocytes (micro m**2)')
    pyplot.ylabel('Frequency')
    pyplot.title(plot_title+" N="+str(len(dict0.keys())))
    pyplot.legend()
    pyplot.savefig(plot_name)
    pyplot.clf()



def plot_three_hists(area_all, area_old_filter, area_new_filter, plot_name_area, plot_name_pp):
    bins = np.linspace(200, 20000, 100)
    pyplot.figure(figsize=(14, 8))
    pyplot.subplot(1, 2, 1)  # First subplot
    pyplot.hist(area_all, bins, alpha=0.5, color="black")
    pyplot.title("before filter: mean="+str(round(stats.mean(area_all),2))+", med="+str(round(stats.median(area_all),2))+", sd="+str(round(stats.stdev(area_all),2))+", N="+str(len(area_all)))
    pyplot.ylabel('counts')
    pyplot.xlabel('size (micrometers**2)')
    pyplot.subplot(1, 2, 2)  # Second subplot
    pyplot.hist(area_filter_in_new, bins, alpha=0.5, color="grey")
    pyplot.hist(area_old_filter, bins, alpha=0.5, color="grey")
    pyplot.ylabel('counts')
    pyplot.xlabel('size (micrometers**2)')
    pyplot.title("after old filter (black):  mean="+str(round(stats.mean(area_old_filter),2))+", med="+str(round(stats.median(area_old_filter),2))+", sd="+str(round(stats.stdev(area_old_filter),2))+", N="+str(len(area_old_filter))+"\n"
                 "after new filter (grey):  mean="+str(round(stats.mean(area_new_filter),2))+", med="+str(round(stats.median(area_new_filter),2))+", sd="+str(round(stats.stdev(area_new_filter),2))+", N="+str(len(area_new_filter)))
    # Adjust spacing between subplots
    pyplot.tight_layout()
    pyplot.savefig(plot_name_area)
    pyplot.clf()
    bins = np.linspace(0, 1, 100)
    pyplot.hist(pp_filter_all, bins)
    pyplot.title("Histogram of PP values")
    pyplot.xlabel("size (micrometers**2)")
    pyplot.ylabel("counts")
    pyplot.savefig(plot_name_pp)
    pyplot.clf()
