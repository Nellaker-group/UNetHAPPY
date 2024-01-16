

def get_pixel_size(cohort):
    pixels = {}
    pixels["leipzig"] = 0.5034
    pixels["munich"] = 0.5034
    pixels["hohenheim"] = 0.5034
    pixels["gtex"] = 0.4942
    pixels["endox"] = 0.2500
    try:
        return(pixels[cohort])
    except KeyError:
        print("Oops!  That cohort has no pixel size.")


def get_which_pixel(indi_id):
    if indi_id.startswith("a"):
        return("leipzig")
    elif indi_id.startswith("m"):
        return("munich")
    elif indi_id.startswith("h"):
        return("hohenheim")
    elif indi_id.startswith("GTEX"):
        return("gtex")
    elif indi_id.startswith("Image"):
        return("endox")
    else:
        print("Not found!")
        return(None)
