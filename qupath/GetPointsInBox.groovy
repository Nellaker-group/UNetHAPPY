"""
Creates a csv of "TAnnot" boxes and cell points ready for generating training data and
annotation files.

You need to manually add the path of where you want them saved in saveDIR
"""
import static qupath.lib.gui.scripting.QPEx.*

// Add your local path for saving here
saveDir = Null

slide_name = getCurrentServer().getPath().split("/")[-1]
slide_number = slide_name.split("-")[0]

// Get all manually annotated box areas
def allBoxAnnot = getAnnotationObjects().findAll({ it.getPathClass().getName() == "TAnnot" })

// Get upper left coord of box and width and height
def xs = allBoxAnnot.collect({ (int) it.getROI().getBoundsX() })
def ys = allBoxAnnot.collect({ (int) it.getROI().getBoundsY() })
def widths = allBoxAnnot.collect({ (int) it.getROI().getBoundsWidth() })
def heights = allBoxAnnot.collect({ (int) it.getROI().getBoundsHeight() })

// Get all cell class points
def points = getAnnotationObjects().findAll({
    it.getPathClass().getName() != "TAnnot"
            && it.getPathClass().getName() != "FAnnot" && it.getPathClass().getName() != "Discuss"
})
def allcyt = points.find({ it.getPathClass().getName() == "CYT" })
def allhof = points.find({ it.getPathClass().getName() == "HOF" })
def allsyn = points.find({ it.getPathClass().getName() == "SYN" })
def allfib = points.find({ it.getPathClass().getName() == "FIB" })
def allven = points.find({ it.getPathClass().getName() == "VEN" })

def allcytInBoxes = []
def allhofInBoxes = []
def allsynInBoxes = []
def allfibInBoxes = []
def allvenInBoxes = []

def getRelativeCoords(cellClassArray, x, y, width, height) {
    // Get the global coordinates of cell classes in the box
    pointsInBox = cellClassArray.getROI().getAllPoints().findAll({
        it.getX() >= x && it.getX() <= (x + width) && it.getY() >= y && it.getY() <= (y + height)
    })
    // Convert these to the coordinates relative to the box
    return pointsInBox.collect({
        [(int) (it.getX() - x), (int) (it.getY() - y)]
    })
};

// Loop through each annotation box and extract the relative coordinates of each cell class in the box
for (int i = 0; i < xs.size(); i++) {
    // Append such that the box indexes and coordinate indexes match
    allcytInBoxes << getRelativeCoords(allcyt, xs[i], ys[i], widths[0], heights[0])
    allhofInBoxes << getRelativeCoords(allhof, xs[i], ys[i], widths[0], heights[0])
    allsynInBoxes << getRelativeCoords(allsyn, xs[i], ys[i], widths[0], heights[0])
    allfibInBoxes << getRelativeCoords(allfib, xs[i], ys[i], widths[0], heights[0])
    allvenInBoxes << getRelativeCoords(allven, xs[i], ys[i], widths[0], heights[0])
};

// Save these to a file with columns boxx, boxy, pointx, pointy, class
def FILE_HEADER = 'bx,by,px,py,class'
def fileName = slide_number + "_from_groovy.csv"
def savePath = saveDir + fileName

def buildRows(sb, boxi, cellArray, x, y, cellName) {
    for (int pointi = 0; pointi < cellArray[boxi].size(); pointi++) {
        sb.append(String.join(',', x.toString(), y.toString(),
                        cellArray[boxi][pointi][0].toString(),
                        cellArray[boxi][pointi][1].toString(), cellName))
        sb.append('\n')
    }
};

// Write to csv
try (PrintWriter writer = new PrintWriter(new File(savePath))) {
    StringBuilder sb = new StringBuilder();
    sb.append(FILE_HEADER)
    sb.append('\n')

    // For each box, write rows for each point
    for (int boxi = 0; boxi < xs.size(); boxi++) {
        buildRows(sb, boxi, allcytInBoxes, xs[boxi], ys[boxi], "CYT")
        buildRows(sb, boxi, allhofInBoxes, xs[boxi], ys[boxi], "HOF")
        buildRows(sb, boxi, allsynInBoxes, xs[boxi], ys[boxi], "SYN")
        buildRows(sb, boxi, allfibInBoxes, xs[boxi], ys[boxi], "FIB")
        buildRows(sb, boxi, allvenInBoxes, xs[boxi], ys[boxi], "VEN")
    }

    print(sb.toString())

    writer.write(sb.toString());
    print("done!")

} catch (FileNotFoundException e) {
    print(e.getMessage())
}
