import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import rasterio.plot
from pyproj import Transformer
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression
import csv








# Datetime to decimal year
def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


# Find the position of the k nearest elements in a 1d array to a value
def find_nearest(array, value, k):
    array = np.asarray(array)
    idx = np.argpartition(np.abs(array - value), tuple(range(0,k)))
    return idx[:k]


# Third order polynomial interpolation
def poly_interp(time, y, deg = 3):   
    model = np.poly1d(np.polyfit(time, y, deg))

    x_pred = np.linspace(time[0], time[-1], int((time[-1]-time[0])*100))
    y_pred = model(x_pred)
    
    return x_pred, y_pred


def reproj(X, Y, source_epsg, target_epsg):
    lon = []
    lat = []
    transformer = Transformer.from_crs(source_epsg, target_epsg, always_xy=True)

    if len(X) == len(Y):
        for i in range(len(X)):
            lon1, lat1 = transformer.transform(X[i], Y[i])
            lon.append(lon1)
            lat.append(lat1)
    else:
        raise ValueError("X and Y must be of the same length.")
    return lon, lat


def get_SERAC_column(SERAC_dic, col_index, dtype = float):
        """
        Get one data column from SERAC dat file using the converted data format
        !!!! TBM: get multiple data column? set limits on col_index?
        """
        col = []
        for key in SERAC_dic:
            col.append(np.array(SERAC_dic[key].iloc[:, col_index].values, dtype = dtype))
        return col


# Total dynamic ice thickness change = total elevation change -  elevation change due to FDM
def total2dynamicH_direct(h_t, h_fdm):
    h_fdm = h_fdm - h_fdm[0]
    h_d = h_t - h_fdm
    return h_d


# Error propation of partitioning
def error_prop_add_direct(err_relh, err_fdm):
    err_fdm_dif = np.sqrt(err_fdm**2+err_fdm[0]**2)
    err = np.sqrt(err_relh**2+err_fdm_dif**2)
    return err


def get_XY_err(X_array, Y_array, ind_ls, coords, dim): ## ind_ls is 2d when dim = 1, 3d when dim = 2  
    x_err = []
    y_err = []
    for i in range(len(ind_ls)):
        x_err1 = []
        y_err1 = []
        for j in range(len(ind_ls[i])):
            if dim == 1:
                x_err1.append(X_array[ind_ls[i][j]] - float(coords[i][0]))
                y_err1.append(Y_array[ind_ls[i][j]] - float(coords[i][0]))
            elif dim == 2:
                x_err1.append(X_array[ind_ls[i][j][0], ind_ls[i][j][1]] - float(coords[i][0]))
                y_err1.append(Y_array[ind_ls[i][j][0], ind_ls[i][j][1]] - float(coords[i][1]))      
            else: 
                raise ValueError("Need to specify the dimension of X_array/Y_array: dim = 1 or 2")
        x_err.append(np.array(x_err1))
        y_err.append(np.array(y_err1))

    mean_x_err = []
    mean_y_err = []
    for i in range(len(ind_ls)):
        mean_x_err.append(np.mean(x_err[i]))
        mean_y_err.append(np.mean(y_err[i]))
    
    return mean_x_err, mean_y_err


def find_nearest_2D(X_array, Y_array, coords, n_n = 4): 
    # For gridded product
    # Number of nearest neighbors to be selected
    ind_ls = []
    if len(X_array) == len(Y_array):  # TBM: coords must be of the same length as well
        for i in range(len(coords)):
            ind_arr = np.argpartition((np.square(X_array - float(coords[i][0])) + np.square(Y_array - float(coords[i][1]))),
                                      tuple(range(0,n_n)))[:n_n]
            ind_ls.append(np.array(ind_arr))
        mean_x_err, mean_y_err = get_XY_err(X_array, Y_array, ind_ls, coords, dim = 1)
    else:
        raise ValueError("X and Y must be of the same length.")
    return ind_ls, mean_x_err, mean_y_err


def find_nearest_withNA(X_array, Y_array, Z_array, coords, k = 10, n_n = 4): # RACMO FDM netcdf file
    # Sort the biggest kth elements in an array. k should be at least equal to n_n, 
    # because the netcdf file of FDM contains grids of NaNs
    # Number of nearest neighbors to be selected
    ind_ls = []
    for i in range(len(coords)): # TBM: coords must be of the same length as well
        ind_fla = np.argpartition((np.square(X_array - float(coords[i][0])) + 
                                   np.square(Y_array - float(coords[i][1]))).flatten(), tuple(range(0,k)))[:k]
        
        ind_near = []
        c = 0
        n = 0
        while n < k:
            ind = divmod(ind_fla[n], X_array.shape[1])
            if not np.isnan(Z_array[:, ind[0], ind[1]]).any():
                ind_near.append(ind)
                c = c+1
                if c == n_n:
                    break # Finish searching after found n_n nearest neighbors without NaNs
            n = n+1
            
        if c < n_n:
            raise ValueError("Only "+str(c)+" nearest neighbors are selected at the "+str(i)+"th location:\n"+
                  str(k-c)+" of the "+str(k)+" nearest neighbors contains NaN.\nPlease increase the value of k.")
        ind_ls.append(np.array(ind_near))
        
    mean_x_err, mean_y_err = get_XY_err(X_array, Y_array, ind_ls, coords, dim = 2)
        
    return ind_ls, mean_x_err, mean_y_err   
    
    
def extract_FDM_TS(fname, coords, n = 4, useCSV = True, plot_n = 0):
    if useCSV:
        FDM_fcsv = pd.read_csv(fname, na_values=-999.0) 
        time = np.array(FDM_fcsv.columns[4:], dtype = float)

        ind_ls, mean_x_err, mean_y_err = find_nearest_2D(FDM_fcsv['X'].values, FDM_fcsv['Y'].values, coords, n_n = n)
    else:
        FDM = xr.open_dataset(fname)
        time = FDM['time'].values
        time_decimal = []
        for t in time:
            time_decimal.append(year_fraction(pd.Timestamp(t).to_pydatetime()))
        time = np.array(time_decimal)
        
        X, Y = reproj(FDM.lon.values.reshape(-1), FDM.lat.values.reshape(-1), "epsg:4326", "epsg:32624")
        X = X.reshape(FDM.lon.values.shape)
        Y = Y.reshape(FDM.lat.values.shape)
        
        ind_ls, mean_x_err, mean_y_err = find_nearest_withNA(X, Y, FDM['zs'].values, coords, n_n = n)

    fdm_serac = []
    fdm_serac_std = []
    for i in range(len(ind_ls)):
        h1 = []
        for j in range(len(ind_ls[i])):
            if useCSV:
                h1.append(FDM_fcsv.iloc[ind_ls[i][j], 4:])
            else:
                h1.append(FDM['zs'].values[:, ind_ls[i][j][0], ind_ls[i][j][1]])
        fdm_serac.append(h1[0].values) ## nearest neighbor
        fdm_serac_std.append(np.std(np.array(h1), axis = 0))
        
    ## Example plot surface height change due to FDM at the kth surface patch
    if isinstance(plot_n, int): # TBM: constraints on the range and data type of plot_n???
        fig, ax = plt.subplots(figsize = (15,5))
        ax.plot(time, fdm_serac[plot_n], c = 'r', label = 'Mean')
        ax.fill_between(time, fdm_serac[plot_n]-fdm_serac_std[plot_n], fdm_serac[plot_n]+fdm_serac_std[plot_n], 
                        facecolor ='grey')
        ax.set_xlabel('Date',fontsize = 13)
        ax.set_ylabel('zs (m)', fontsize=13)
        ax.set_title('surface height due to SMB and firn compaction relative to (1960-1979)', fontsize=15)
        ax.legend()
        ax.grid(True)

    return time, fdm_serac, fdm_serac_std, mean_x_err, mean_y_err


def resample_FDM(time, fdm_serac, fdm_serac_std, df_SERAC): ## temporal resample of FDM to the time of SERAC
    time_fdm = []
    time_err = []
    h_fdm = []
    h_fdm_std = []
    ind = 0
    for key in df_SERAC:
        df_1pts = df_SERAC[key]
        t_ind = []
        for i in range(len(df_1pts)):
            t = df_1pts.iloc[i, 1]
            t_ind_1meas = find_nearest(time, t, 1).item()
            t_ind.append(t_ind_1meas)
        time_fdm.append(time[t_ind])
        time_err.append(time[t_ind] - df_1pts.iloc[:, 1].values)
        h_fdm.append(fdm_serac[ind][t_ind])
        h_fdm_std.append(fdm_serac_std[ind][t_ind])
        ind += 1
        
    return time_fdm, time_err, h_fdm, h_fdm_std













class Read:
    
    def __init__(self, fname = None, skiprows = None, max_col_count = 10):
        self.fname = fname
        
        df = pd.read_table(self.fname, sep = "\s+", skiprows = skiprows, header = None, 
                           names=range(max_col_count)).reset_index(drop=True)

        header = df[df[0].astype(str).str.isnumeric()].copy()
        header.iloc[:, 1] = np.array(header.iloc[:, 1].values, dtype = int)
        header.iloc[:, -1] = np.array(header.iloc[:, -1].values, dtype = int)

        self.df = df.drop(header.index).reset_index(drop = True).dropna(axis = 1)
        # df[1] = ['ICESat' if item[1].isnumeric() else item for item in df[1]]
        self.header = header.reset_index(drop = True).dropna(axis = 1)    
        patch_index = np.insert(np.cumsum(self.header[1].values), 0, 0)
        
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

    
    
class Partition:
    def __init__(self, header = None, SERAC_dic = None):
        
        header['lon'], header['lat'] = reproj(header.iloc[:, 3], header.iloc[:, 4], "epsg:32624", "epsg:4326")
        
        self.header = header
        self.SERAC_dic = SERAC_dic
    
        
    def detect_outlier(self): ## Add parameters to be more flexible
        ## Detect outlier and create flags
        time = get_SERAC_column(self.SERAC_dic, 1)
        rel_h = get_SERAC_column(self.SERAC_dic, 4)

        outlier_flag = []
        for i in range(len(rel_h)):
            x_pred, y_pred = poly_interp(time[i], rel_h[i], 3)
            interp_ts = pd.Series(data=y_pred, index=np.round(x_pred, 2))
            interp_ts = interp_ts.reindex(time[i], method = 'nearest')
            resid = rel_h[i] - interp_ts.values
            outlier_msk_mad = (np.abs(resid - np.median(resid))/np.median(np.abs(resid - np.median(resid)))) < 3.5
            outlier_msk_std = np.abs(resid - np.median(resid)) < 3*np.std(resid)

            # Outliers are those whose residuals of polynomial interpolation are outside of the 3 standard deviations from the median of the residuals
            outlier_flag.append(outlier_msk_std) 
        
        return outlier_flag # True: good data; False: outliers
      
    
    def get_geoid_height(self, f_geoid, plot_geoid = False):
        ## Read geoid data and add geoid height to header
        geoid_GIS= rasterio.open(f_geoid)
        geoid = geoid_GIS.read(1)
        
        if plot_geoid:
            rasterio.plot.show(geoid, transform=geoid_GIS.transform, cmap='viridis', vmin = 0, aspect = 'auto')
            plt.show()

        ## Get geoid coordinates and find the geoid heights at locations of SERAC surface patches

        lon_geoid = []
        for i in range(geoid_GIS.width):
            lon_geoid.append(geoid_GIS.xy(0,i)[0])
        lat_geoid = []
        for i in range(geoid_GIS.height):
            lat_geoid.append(geoid_GIS.xy(i,0)[1])

        geoid_SERAC = []
        for j in range(len(self.header)):
            ind_lat = find_nearest(lat_geoid, self.header['lat'][j], 1).item()
            ind_lon = find_nearest(lon_geoid, self.header['lon'][j], 1).item()
            # The geoid has been shifted by -15.6 cm from the global WGS84 computation system
            geoid_SERAC.append(geoid[ind_lat,ind_lon]+0.156)

        self.header['geoid'] = geoid_SERAC
        
        return self.header
    
    
    def flagging(self, outlier_flag = None, threshold = 10, plot_code = False): 
        """
        Create flags: 1 = good data, 2 = outlier, 3 = good data, but at sea-level, 
        4 = outlier at sea-level 
        (could be false outlier as the polynomial interpolation can't catch the drop of surface elevation to sea-level)
        """
        time = get_SERAC_column(self.SERAC_dic, 1)
        absh = get_SERAC_column(self.SERAC_dic, 6)
        err_relh = get_SERAC_column(self.SERAC_dic, 5)

        flags = []
        for i in range(len(absh)):
            flag = outlier_flag[i]*1
            flag[flag==0] = 2

            if np.any(absh[i]<(self.header['geoid'][i]+threshold)): # Measurements within 10 m of GEOID height 
                sl_msk = (absh[i]<(self.header['geoid'][i]+threshold))*1
                flags.append(flag + sl_msk*2)
            else:
                flags.append(flag)
        
        self.flags = flags
        
        if plot_code: ## add ways to plot bigger than 1? constrain how many plots can be shown?
            for i in range(len(absh)):
                if np.any(flags[i]==plot_code):
                    fig = plt.figure(figsize = (15,5))
                    ax1 = fig.add_subplot(1, 1, 1)
                    ax1.scatter(time[i], absh[i], c = 'red')
                    plt.errorbar(time[i], absh[i], yerr = err_relh[i], color='grey', linestyle='')
                    ax1.set_xticks(np.arange(int(time[i][0]), int(time[i][-1]+2), 2))
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.set_title(header[0][i]+' GEOID height: '+str(np.round(header['geoid'][i], 2)), fontsize = 22)
                    ax1.set(xlabel=None, ylabel=None)

        return flags    
        
        
    def partition_FDM(self, FDM_fname, useCSV, n, plot_n):
        coords = np.array([self.header[3].values, self.header[4].values]).T
        time, fdm_serac, fdm_serac_std, mean_x_err, mean_y_err = extract_FDM_TS(FDM_fname, coords, 
                                                                                n = 4, useCSV = useCSV, plot_n = plot_n)

        time_fdm, time_err, h_fdm, h_fdm_std = resample_FDM(time, fdm_serac, fdm_serac_std, self.SERAC_dic)
        
        ## Get Surface elevation relative to reference elevation (m) (rel_h) 
        rel_h = get_SERAC_column(self.SERAC_dic, 4)
        err_rel_h = get_SERAC_column(self.SERAC_dic, 5)

        ## Calculate dynamic ice thickness change and errors

        d_h_result = []
        err_d_h = []
        for j in range(len(rel_h)):
            d_h_result.append(total2dynamicH_direct(rel_h[j], h_fdm[j]))
            err_d_h.append(error_prop_add_direct(err_rel_h[j], h_fdm_std[j]))

        self.mean_x_err = mean_x_err
        self.mean_y_err = mean_y_err
        self.time_err = time_err
        self.h_fdm = h_fdm
        self.h_fdm_std = h_fdm_std
        self.d_h_result = d_h_result
        self.err_d_h = err_d_h

        return h_fdm, h_fdm_std, d_h_result, err_d_h
        
        
    def save_txt(self, output_fname, add_flag = True):
        ## Prepare for saving data to text file
        ind = 0
        for key in self.SERAC_dic:
            d_1patch = np.array(self.SERAC_dic[key])

            ## Add height change due to FDM at the closest grid with std from the 4 nearest neighbors, 
            ## and dynamic ice thickness change and formally propagated errors to SERAC data array
            h_fdm_d = np.round(np.array([self.h_fdm[ind], self.h_fdm_std[ind], self.d_h_result[ind], self.err_d_h[ind]]).T, 3)
            d_1patch = np.insert(d_1patch, [d_1patch.shape[1]], h_fdm_d, axis=1)

            if add_flag:
                d_1patch = np.insert(d_1patch, [d_1patch.shape[1]], np.reshape(self.flags[ind], (-1, 1)), axis=1)

            ## Format the data to the same precision in SERAC reconstruction
            for k in range(len(d_1patch)):
                for m in [7, 8]:
                    d_1patch[k,m] = int(d_1patch[k,m])

            self.SERAC_dic[key] = d_1patch
            ind += 1

        ## Save to text file
        with open(output_fname, 'w', encoding='UTF8') as f:
            for i in range(len(self.header)):
                # write the header line
                f.write("\t".join(str(item) for item in self.header.iloc[i,:].values))
                f.write("\n")

                # write SERAC records
                for j in range(len(self.SERAC_dic[self.header[0][i]])):
                    f.write(" ".join("{: >12}".format(str(item)) for item in self.SERAC_dic[self.header[0][i]][j]))
                    f.write("\n")



    
       