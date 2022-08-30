import numpy as np
import pandas as pd






class Read:
    
    def __init__(self, fpath = None, tile_id = None):
        self.fpath = fpath
        
        if tile_id not in range(6, 8):
            raise ValueError("Please provide a valid tile id.")
        else:
            self.tile_id = tile_id
            self.fname = "TimeSeriesTile"+str(self.tile_id)+"_partitioned.txt"
        
        df = pd.read_table(self.fpath+self.fname, sep = "\s+").reset_index(drop=False)
        row1 = np.array(df.columns)
        df0 = pd.DataFrame(data = row1.reshape(1,-1))
        df = df.reset_index(drop = False)
        df.columns = range(df.columns.size)
        df = pd.concat([df0 ,df]).reset_index(drop = True)
        df = df.drop(columns=[0])
        
        header = df[df[1].astype(str).str.isnumeric()]
        for i in range(len(header)):
            header.iloc[i,1] = int(header.iloc[i,1])     

        self.df = df.drop(header.index).reset_index(drop = True)
        # df[1] = ['ICESat' if item[1].isnumeric() else item for item in df[1]]
        self.header = header.reset_index(drop = True).dropna(axis = 1)

        patch_index = np.cumsum(self.header[2].values)
        patch_index = np.insert(patch_index, 0, 0)

        SERAC_dic = {}
        for k in range(len(self.header)):
            SERAC_dic[self.header.iloc[k,0]] = self.df[patch_index[k]:patch_index[k+1]]
        self.SERAC_dic = SERAC_dic
        
    def return_raw(self): 
        """
        Return the header information and SERAC raw data
        !!!! TBM: not sure, is it appropriate to return the dictionary?
        """
        return self.header, self.SERAC_dic

    
    def get_SERAC_column(self, col_index):
        """
        Get one data column from SERAC dat file using the converted data format
        !!!! TBM: get multiple data column? set limits on col_index?
        """
        h = []
        for key in self.SERAC_dic:
            h.append(self.SERAC_dic[key].iloc[:, col_index].values)
        return h
        
        
        