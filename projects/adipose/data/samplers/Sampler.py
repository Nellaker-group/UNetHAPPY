import torch
import math
import numpy as np
from torch.utils.data.sampler import Sampler

class MontageSamplerUniform(Sampler):
    """Samples tiles randomly from montages, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            x0, y0 = (0,0)
            rand_var = np.random.randint(0,len(self.data_source.input_images))
            H, W=self.data_source.input_images[rand_var].shape
            # to make sure we do not get np.random.randint(0, 0) which gives an error
            rangeH=max(H - wdw_H,1)
            rangeW=max(W - wdw_W,1)
            x0, y0 = np.random.randint(0, rangeH), np.random.randint(0, rangeW)
            listie.append((rand_var,x0,y0,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source)
    
    
    
    
class MontageSamplerUniformFrankenstein(Sampler):
    """Samples 4 pieces of tile randomly from montages and stitches them together to one tile, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            xs = []
            ys = []
            rand_var = []
            x0, y0 = (0,0)
            # for getting them 4 parts
            for i in range(4):
                rand_var.append(np.random.randint(0,len(self.data_source.input_images)))
                H, W=self.data_source.input_images[rand_var[i]].shape
                # to make sure we do not get np.random.randint(0, 0) which gives an error
                rangeH=max(H - wdw_H,1)
                rangeW=max(W - wdw_W,1)
                # to slice and dice
                xs.append(np.random.randint(0, rangeH))
                ys.append(np.random.randint(0, rangeW))
                cutx, cuty = np.random.randint(0, wdw_H), np.random.randint(0, wdw_W)
            listie.append((rand_var,(xs,ys),(cutx,cuty),first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class MontageSamplerDatasetSize(Sampler):
    """Samples tiles randomly from montages, but taking size of each montage into account, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    # so this has to have the data
    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)

        index = 0
        tiles = []
        for e in self.data_source.input_images:
            H, W=self.data_source.input_images[index].shape
            tiles.append(math.floor(H//wdw_W)*math.floor(W//wdw_W))
            index += 1
            
        tilesAll = tiles.sum()
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            x0, y0 = (0,0)
            rand_var = np.random.randint(0,tilesAll)
            whichDataset = -9
            H, W=self.data_source.input_images[rand_var].shape
            # to make sure we do not get np.random.randint(0, 0) which gives an error
            rangeH=max(H - wdw_H,1)
            rangeW=max(W - wdw_W,1)
            # to make sure we do not get np.random.randint(0, 0) which gives an error
            for i in range(len(tiles)):
                if i == 0 and rand_var < tiles[i]:
                    x0, y0 = np.random.randint(0, rangeH), np.random.randint(0, rangeW)
                    whichDataset = i
                elif rand_var >= tiles[i-1] and rand_var < tiles[i]:
                    x0, y0 = np.random.randint(0, rangeH), np.random.randint(0, rangeW)
                    whichDataset = i
            assert whichDataset >= 0
            listie.append((whichDataset,x0,y0,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class SeqSamplerUniform(Sampler):
    """Samples tiles randomly, not according to premade tiles, with replacement, but uniformly across datasets.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method

    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)

        first = 1 
        listie=[]

        for sample_idx in range(self.sample_size):

            x0, y0 = (0,0)
            rand_var = np.random.randint(0,len(self.data_source.input_images))
            index=-1
            H, W=self.data_source.input_images[rand_var].shape
            # to make sure we do not get np.random.randint(0, 0) which gives an error 
            rangeH=max(H - wdw_H,1)
            rangeW=max(W - wdw_W,1)
            x0, y0 = np.random.choice(list(range(0, (rangeH+1), wdw_W))), np.random.choice(list(range(0, (rangeW+1), wdw_W)))
            index = rand_var
            listie.append((index,x0,y0,first))
            first = 0
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class SeqSamplerDatasetSize(Sampler):
    """Samples whole/sequantial tiles randomly, sampling premade tiles as they are, with replacement, takes size of the datasets into account when sampling.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method

    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)

        index = 0
        tiles = []
        for e in self.data_source.input_images:
            H, W=self.data_source.input_images[index].shape
            tiles.append(math.floor(H//wdw_W)*math.floor(W//wdw_W))
            index += 1

        tilesAll = tiles.sum()
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):
            x0, y0 = (0,0)
            rand_var = np.random.randint(0,tilesAll)
            whichDataset = -9
            # to make sure we do not get np.random.randint(0, 0) which gives an error                                               
            H, W=self.data_source.input_images[rand_var].shape
            rangeH=max(H - wdw_H,1)
            rangeW=max(W - wdw_W,1) 
            for i in range(len(tiles)):
                if i == 0 and rand_var < tiles[i]:
                    x0, y0 = np.random.choice(list(range(0, (rangeH+1), wdw_W))), np.random.choice(list(range(0, (rangeW+1), wdw_W)))
                    whichDataset = i
                elif rand_var >= tiles[i-1] and rand_var < tiles[i]:
                    x0, y0 = np.random.choice(list(range(0, (rangeH+1), wdw_W))), np.random.choice(list(range(0, (rangeW+1), wdw_W)))
                    whichDataset = i
            assert whichDataset >= 0
            listie.append((whichDataset,x0,y0,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class ImageSamplerUniform(Sampler):
    """Samples images randomly from folders, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            whichData = np.random.randint(0,len(self.data_source.input_images))
            whichImage = np.random.randint(0,len(self.data_source.input_images[whichData]))
            listie.append((whichData,whichImage,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source.input_images)
    

class ImageSamplerUniformFrankenstein(Sampler):
    """Samples 4 pieces of tile randomly from four images and stitches them together to one tile, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            xs = []
            ys = []
            rand_var = []
            # for getting them 4 parts            
            whichData = np.random.randint(0,len(self.data_source.input_images))
            whichImage = np.random.randint(0,len(self.data_source.input_images[whichData]))
            whichData2 = np.random.randint(0,len(self.data_source.input_images))
            whichImage2 = np.random.randint(0,len(self.data_source.input_images[whichData2]))
            whichData3 = np.random.randint(0,len(self.data_source.input_images))
            whichImage3 = np.random.randint(0,len(self.data_source.input_images[whichData3]))
            whichData4 = np.random.randint(0,len(self.data_source.input_images))
            whichImage4 = np.random.randint(0,len(self.data_source.input_images[whichData4]))
            # to slice and dice
            cutx, cuty = np.random.randint(0, wdw_W), np.random.randint(0, wdw_W)
            whichDataTuple = (whichData,whichData2,whichData3,whichData4)
            whichImageAndCut = ((whichImage,whichImage2,whichImage3,whichImage4),(cutx,cuty)) 
            listie.append((whichDataTuple,whichImageAndCut,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source.input_images)



class ImageSamplerUniformFrankensteinEmil(Sampler):
    """Samples 4 pieces of tile randomly from four images from the same dataset and stitches them together to one tile, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            xs = []
            ys = []
            rand_var = []
            # for getting them 4 parts            
            whichDataset = np.random.randint(0,len(self.data_source.input_images))
            whichImage = np.random.randint(0,len(self.data_source.input_images[whichDataset]))
            whichImage2 = np.random.randint(0,len(self.data_source.input_images[whichDataset]))
            whichImage3 = np.random.randint(0,len(self.data_source.input_images[whichDataset]))
            whichImage4 = np.random.randint(0,len(self.data_source.input_images[whichDataset]))
            # to slice and dice
            cutx, cuty = np.random.randint(0, wdw_W), np.random.randint(0, wdw_W)
            whichDataTuple = (whichDataset,whichDataset,whichDataset,whichDataset)
            whichImageAndCut = ((whichImage,whichImage2,whichImage3,whichImage4),(cutx,cuty)) 
            listie.append((whichDataTuple,whichImageAndCut,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source.input_images)



class ImageSamplerUniformFrankensteinV2(Sampler):
    """Samples 4 pieces of tile randomly from four images and stitches them together to one tile, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            xs = []
            ys = []
            rand_var = []
            # for getting them 4 parts            
            whichData0 = np.random.randint(0,len(self.data_source.input_images))
            whichImage0 = np.random.randint(0,len(self.data_source.input_images[whichData0]))
            whichData2 = np.random.randint(0,len(self.data_source.input_images))
            whichImage2 = np.random.randint(0,len(self.data_source.input_images[whichData2]))
            whichData3 = np.random.randint(0,len(self.data_source.input_images))
            whichImage3 = np.random.randint(0,len(self.data_source.input_images[whichData3]))
            whichData4 = np.random.randint(0,len(self.data_source.input_images))
            whichImage4 = np.random.randint(0,len(self.data_source.input_images[whichData4]))

            h = np.random.randint(0,wdw_W)
            w = np.random.randint(0,wdw_W)
            w1 = np.random.randint(0,wdw_W)

            tile0 = (h,w)
            tile1 = (h,(wdw_W-w))
            tile2 = ((wdw_W-h),w1)
            tile3 = ((wdw_W-h),(wdw_W-w1))

            # to slice and dice
            x0, y0 = (wdw_H,wdw_W)
            while(tile0[0] > (wdw_W-x0) or tile0[1] > (wdw_W-y0)):
                x0, y0 = np.random.randint(0, wdw_W), np.random.randint(0, wdw_W)
            x1, y1 = (wdw_H,wdw_W)
            while(tile1[0] > (wdw_W-x1) or tile1[1] > (wdw_W-y1)):
                x1, y1 = np.random.randint(0, wdw_W), np.random.randint(0, wdw_W)
            x2, y2 = (wdw_H,wdw_W)
            while(tile2[0] > (wdw_W-x2) or tile2[1] > (wdw_W-y2)):
                x2, y2 = np.random.randint(0, wdw_W), np.random.randint(0, wdw_W)
            x3, y3 = (wdw_H,wdw_W)
            while(tile3[0] > (wdw_W-x3) or tile3[1] > (wdw_W-y3)):
                x3, y3 = np.random.randint(0, wdw_W), np.random.randint(0, wdw_W)

            whichDataTuple = (whichData0,whichData2,whichData3,whichData4)
            whichImageAndCut = ((whichImage0,whichImage2,whichImage3,whichImage4),(x0,y0,x1,y1,x2,y2,x3,y3,h,w,w1)) 
            listie.append((whichDataTuple,whichImageAndCut,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source.input_images)


class ImageSamplerDatasetSize(Sampler):
    """Samples tiles randomly from montages, but taking size of each montage into account, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    # so this has to have the data
    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method
    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)

        index = 0
        tiles = []
        for e in self.data_source.input_images:
            tiles.append(len(self.data_source.input_images[index]))
            index += 1
            
        tilesAll = tiles.sum()
        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            rand_var = np.random.randint(0,tilesAll)
            whichData = -9
            # to make sure we do not get np.random.randint(0, 0) which gives an error
            for i in range(len(tiles)):
                if i == 0 and rand_var < tiles[i]:
                    whichImage = np.random.randint(0,len(self.data_source.input_images[whichData]))
                    whichData = i
                elif rand_var >= tiles[i-1] and rand_var < tiles[i]:
                    whichImage = np.random.randint(0,len(self.data_source.input_images[whichData]))
                    whichData = i
            assert whichDataset >= 0
            listie.append((whichData,whichImage,first))
            first = 0
            self.counter += 1
        return(iter(listie))

    def __len__(self):
        return len(self.data_source)



class ValSampler(Sampler):
    """Samples all tiles from validation
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, shape, counter):

        self.data_source = data_source
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method

    def __iter__(self):        

        wdw_H, wdw_W = (self.shape,self.shape)

        tiles = []
        for index in range(len(self.data_source.input_images)):
            H, W=self.data_source.input_images[index].shape
            tiles.append(math.floor(H//wdw_H)*math.floor(W//wdw_W))

        first = 1 
        listie=[]

        counter = [0]*len(self.data_source.input_images)

        for i in range(len(self.data_source.input_images)):
            H, W=self.data_source.input_images[i].shape
            # to make sure we do not get np.random.randint(0, 0) which gives an error
            rangeH=max(H - wdw_H+1,1)
            rangeW=max(W - wdw_W+1,1)
            for x in range(0, rangeH, wdw_H):
                for y in range(0, rangeW, wdw_W):
                    listie.append((i,x,y,first))
                    first = 0
                    counter[i] += 1

        for i in range(len(counter)):
            assert tiles[i] == counter[i]
        
        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class ImageValSampler(Sampler):
    """Samples all images from validation
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, shape, counter):

        self.data_source = data_source
        self.shape = shape
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method

    def __iter__(self):        

        first = 1 
        listie=[]
        counter = [0]*len(self.data_source.input_images)
        for i in range(len(self.data_source.input_images)):
            for j in range(len(self.data_source.input_images[i])):
                    listie.append((i,j,first))
                    first = 0
                    counter[i] += 1

        for i in range(len(counter)):
            assert len(self.data_source.input_images[i]) == counter[i]
        
        return(iter(listie))

    def __len__(self):
        return len(self.data_source.input_images)
