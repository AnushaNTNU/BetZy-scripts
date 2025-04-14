# %%
import xarray as xr
import numpy as np
import glob as glob
import pandas as pd
import sparse 
import os

# %% [markdown]
# ##read the cdo merged h1 output files from the 70 year simulation or 1 year simulation(monthly) concatenated using cdo

# %%
#filename = '/cluster/work/users/a2021/archive/NSSP534frc2esm_f19_tn14_BECCS_land_190124/lnd/hist/All_NSSP534frc2esm_f19_tn14_BECCS_land_190124.clm2.h1_2061_2100.nc' ###BECCS
filename = '/cluster/work/users/a2021/archive/NSSP534frc2esm_f19_tn14_BECCS_OAE_2030high_260124/lnd/hist/All_NSSP534frc2esm_f19_tn14_BECCS_OAE_2030high_260124.clm2.h1.2061-2100.nc' ###sBECCS-OAE

data = xr.open_dataset(filename, decode_times=False)
data

# %%
filename = '/cluster/projects/nn9576k/anusha/DATA/gridarea_192_288.nc'      #default
gridarea = xr.open_dataset(filename, decode_times=False)

# %% [markdown]
# ##convert the h1 sparse data to normal gridded nc data

# %%
for var_name in data.variables:
        if data[var_name].dtype == 'int32':
            data[var_name] = data[var_name].astype('float64')
pft_constants = xr.open_dataset("/cluster/shared/noresm/inputdata/lnd/clm2/paramdata/clm5_params.c171117.nc")
pftnames = pft_constants.pftname
def to_sparse(data, vegtype, jxy, ixy, shape):
    # This constructs a list of coordinate locations at which data exists
    # it works for arbitrary number of dimensions but assumes that the last dimension
    # is the "stacked" dimension i.e. "pft"
    if data.ndim == 1:
        coords = np.stack([vegtype, jxy - 1, ixy - 1], axis=0)
    elif data.ndim == 2:
        # generate some repeated time indexes
        # [0 0 0 ... 1 1 1... ]
        itime = np.repeat(np.arange(data.shape[0]), data.shape[1])
        # expand vegtype and friends for all time instants
        # by sequentially concatenating each array for each time instants
        tostack = [np.concatenate([array] * data.shape[0]) for array in [vegtype, jxy - 1, ixy - 1]]
        coords = np.stack([itime] + tostack, axis=0)
    else:
        raise NotImplementedError

    return sparse.COO(
        coords=coords,
        data=data.ravel(),
        shape=data.shape[:-1] + shape,
        fill_value=np.nan,
    )


def convert_pft_variables_to_sparse(dataset, pftnames):
    # extract PFT variables
    pfts = xr.Dataset({k: v for k, v in dataset.items() if "pft" in v.dims})

    # extract coordinate index locations
    ixy = dataset.pfts1d_ixy.astype(int)
    jxy = dataset.pfts1d_jxy.astype(int)
    vegtype = dataset.pfts1d_itype_veg.astype(int)
    npft = len(pftnames.data)

    # expected shape of sparse arrays to pass to `to_sparse` (excludes time)
    output_sizes = {
        "vegtype": npft,
        "lat": dataset.sizes["lat"],
        "lon": dataset.sizes["lon"],
    }

    result = xr.Dataset()
    # we loop over variables so we can specify the appropriate dtype
    for var in pfts:
        result[var] = xr.apply_ufunc(
            to_sparse,
            pfts[var],
            vegtype,
            jxy,
            ixy,
            kwargs=dict(shape=tuple(output_sizes.values())),
            input_core_dims=[["pft"]] * 4,
            output_core_dims=[["vegtype", "lat", "lon"]],
            dask="parallelized",
            dask_gufunc_kwargs=dict(
                meta=sparse.COO(np.array([], dtype=pfts[var].dtype)),
                output_sizes=output_sizes,
            ),
            keep_attrs=True,
        )

    # copy over coordinate variables lat, lon
    result = result.update(dataset[["lat", "lon"]])
    result["vegtype"] = pftnames.data
    # save the dataset attributes
    result.attrs = dataset.attrs
    return result

# %%
sparse_data = convert_pft_variables_to_sparse(data, pftnames)


# %% [markdown]
# ##first write the output NPP in gc/m2/sec to gridded nc data 

# %%
##AGAGNPP for sugarcane in gC/m²/sec
selected_vegtypes = [67,68]
AGNPP_cane =(sparse_data.AGNPP.isel(vegtype=selected_vegtypes)*sparse_data.pfts1d_wtgcell.isel(vegtype=selected_vegtypes)).sum('vegtype') 
AGNPP_cane_nc = AGNPP_cane.as_numpy()
AGNPP_cane_nc = xr.where(AGNPP_cane_nc >= 0, AGNPP_cane_nc, 0)
AGNPP_cane_nc
AGNPP_cane_nc.name= "AGNPP"
AGNPP_cane_nc.attrs['units'] = "gC/m^2/s"
AGNPP_cane_nc.attrs['cell_methods'] = "time: mean"
AGNPP_cane_nc.attrs['_FillValue'] = 1.e+36
AGNPP_cane_nc.attrs['missing_value'] = 1.e+36
AGNPP_cane_nc.attrs['long_name'] = "net primary production"
#AGNPP_cane_nc.to_netcdf('/cluster/projects/nn9576k/anusha/DATA/AGNPP_NSSP534frc2esm_f19_tn14_BECCS_land_190124.clm2.h1.2061_2100_gridded.nc', mode='w', format='NETCDF4')
AGNPP_cane_nc.to_netcdf('/cluster/projects/nn9576k/anusha/DATA/NSSP534frc2esm_f19_tn14_BECCS_OAE_2030high_260124.clm2.h1.2061-2100_gridded.nc', mode='w', format='NETCDF4')




