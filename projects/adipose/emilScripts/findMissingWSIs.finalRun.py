import os
import sys
# to be able to read in the libaries properly
sys.path.append(os.getcwd())

from db.slides import Slide
import db.eval_runs_interface as db
from peewee import fn


files1 = os.listdir("littleLogFiles_04apr_finalRun/")
files2 = os.listdir("littleLogFiles_24apr_finalRunRestOfSlides/")
realFiles1 =  [f.split("_id")[0] for f in files1]
realFiles2 =  [f.split("_id")[0] for f in files2]

print(realFiles1[0:50])
print(realFiles2[0:50])

realFilesTMP = [realFiles1,realFiles2]
realFilesTMP2 = [item for sublist in realFilesTMP for item in sublist]
realFiles = list(set(realFilesTMP2))


leipzig = 0
munich = 0 
hohenheim = 0
gtex = 0
endox = 0

for realFile in realFiles:
    if realFile.startswith("a"):
        leipzig += 1
    elif realFile.startswith("m"):
        munich += 1
    elif realFile.startswith("h"):
        hohenheim += 1
    elif realFile.startswith("GTEX"):
        gtex += 1
    elif realFile.startswith("Image"):
        endox += 1

db.init("main_24apr.db")


missLeipzig = 0
missMunich = 0 
missHohenheim = 0
missGtex = 0
missEndox = 0

missingSlideIDs = []

# returns the path to a slide
rows = Slide.select().count()
for slide_id in range(rows):
    slide = Slide.get_by_id(slide_id+1)
    slideName = slide.slide_name.split("/")[-1]
    if not slideName in realFiles:
        print(slideName)
        print(slide_id+1)
        print("")
        missingSlideIDs.append(slide_id+1)
        if slideName.startswith("a"):
            missLeipzig += 1
        elif slideName.startswith("m"):
            missMunich += 1
        elif slideName.startswith("h"):
            missHohenheim += 1
        elif slideName.startswith("GTEX"):
            missGtex += 1
        elif slideName.startswith("Image"):
            missEndox += 1


print("missingSlides")
print(missingSlideIDs)

print("missLeipzig/leipzig")
print((missLeipzig/leipzig)*100)

print("missMunich/munich")
print((missMunich/munich)*100)

print("missHohenheim/hohenheim")
print((missHohenheim/hohenheim)*100)

print("missGtex/gtex")
print((missGtex/gtex)*100)

print("missEndox/endox")
print((missEndox/endox)*100)


