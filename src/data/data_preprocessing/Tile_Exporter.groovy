/**
 Final Script For Creating Tiles of Whole Slide Images Using Qupath 
 */

def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def outputPath = buildFilePath('/Users/bermanlab/Desktop/Pranav_Kataria/Image-Tiles-20X', name)
mkdirs(outputPath)

double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = 1
double requestedPixelSize = pixelSize*downsample
//print downsample
//print pixelSize

new TileExporter(imageData)
    .downsample(downsample)   
    .imageExtension('.png')   // .tif file format stores the metadata 
    .tileSize(1000)            // Define size of each tile, in pixels
    .annotatedTilesOnly(false) // If true, only export tiles if there is a (classified) annotation present
    .overlap(0)              // Define overlap, in pixel units at the export resolution
    .writeTiles(outputPath)   // Write tiles to the specified directory

print 'Tiling Completed For the Image'