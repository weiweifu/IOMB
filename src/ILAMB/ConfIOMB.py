"""The data used in this confrontation is queried from USGS servers. As part of the comparison, we compute two metrics to        
    measure performance:

1. Nash-Sutcliffe Efficiency (NSE)

    ``NSE = 1 - SUM((mod - ref)^2) / SUM(ref - MEAN(ref))^2``

2. Kling-Gupta Efficiency (KGE)

    ``KGE = 1 - SQRT( (CORR(ref,mod)-1)^2 + (STD(mod)/STD(ref)-1)^2 + (MEAN(mod)/MEAN(ref)-1)^2 )``

"""
import os
import re
from   functools import partial
from   typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from   cf_units import Unit
from   netCDF4 import Dataset

from   ILAMB import Post as post
from   ILAMB.Confrontation import Confrontation
from   ILAMB import ilamblib as il
from   ILAMB.Variable import Variable

def _depth2levitus(lev,data,kind='nearest'):
  '''
  from original depths given in 'lev' 
  interploated vertically to standard 
  Levitus depths (33)
  lev: 1d array
  data: 3d or 4d (time,lev,lat,lon) or (lev,lat,lon)
  '''
  from scipy.interpolate import interp1d

  slev,sdep,lev_bnd = _levbnd()

  data = np.ma.filled(data,np.nan)

  f = interp1d(lev,data,kind=kind,fill_value='extrapolate',axis=-3)
  d = f(slev)
  # data_new = np.ma.masked_where(d==0,d)
  data_new = np.ma.masked_invalid(d)
  # data_new = d
  return data_new

def _levbnd():
  '''
  define the standard levitus levels
  in the vertical for data interpolation
  '''
  slev = np.array([   0,   10,   20,   30,   50,   75,  100,  125,  150,  200,  250,
                    300,  400,  500,  600,  700,  800,  900, 1000, 1100, 1200, 1300,
                  1400, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])
  lev1 = [   0. ,    5. ,   15. ,   25. ,   40. ,   62.5,   87.5,
                    112.5,  137.5,  175. ,  225. ,  275. ,  350. ,  450. ,
                    550. ,  650. ,  750. ,  850. ,  950. , 1050. , 1150. ,
                   1250. , 1350. , 1450. , 1625. , 1875. , 2250. , 2750. ,
                   3250. , 3750. , 4250. , 4750. , 5250. ]
  lev2 = [   5. ,   15. ,   25. ,   40. ,   62.5,   87.5,  112.5,
                    137.5,  175. ,  225. ,  275. ,  350. ,  450. ,  550. ,
                    650. ,  750. ,  850. ,  950. , 1050. , 1150. , 1250. ,
                   1350. , 1450. , 1625. , 1875. , 2250. , 2750. , 3250. ,
                   3750. , 4250. , 4750. , 5250. , 5500. ]
  sdep = np.asarray(lev2) - np.asarray(lev1)
  lev_bnd = np.asarray([lev1,lev2]).T
  return slev, sdep, lev_bnd

class ConfIOMB(Confrontation):
    """A confrontation for ocean model analysis against 2D or 3D data in global/regional ocean."""

    def __init__(self, **keywords):
        # Ugly, but this is how we call the Confrontation constructor
        super(ConfIOMB, self).__init__(**keywords)  # old 2.7 style of super()

        # Now we overwrite some things which are different here
        # self.regions = ["global"]
        # self.layout.regions = self.regions


    def stageData(self, m):
        r"""Extracts model data and interpolate in the vertical to match the confrontation dataset.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model result context

        Returns
        -------
        obs : ILAMB.Variable.Variable
            the variable context associated with the observational dataset
        mod : ILAMB.Variable.Variable
            the variable context associated with the model result

        """
        # for now we use the defulat stageData
        # obs, mod = super(ConfIOMB, self).stageData(m)
        self.extents = np.asarray([[-90.0, +90.0], [0.0, +360.0]])
        # print('now in ConfIOMB stageData beginning',self.extents)
        # print('depth type is ',type(self.keywords['depth']),'self has depth',self.keywords['depth'] )
        obs = Variable(
            filename=self.source,
            variable_name=self.variable,
            alternate_vars=self.alternate_vars,
            t0=None if len(self.study_limits) != 2 else self.study_limits[0],
            tf=None if len(self.study_limits) != 2 else self.study_limits[1],
        )
        obs.data *= self.scale_factor
        if obs.time is None:
            raise il.NotTemporalVariable()
        self.pruneRegions(obs)

        # The reference might be layered and we want to extract a
        # slice to compare against models
        if "depth" in self.keywords and obs.layered:
            obs.trim(d=[float(self.keywords["depth"]) - 0.01, float(self.keywords["depth"]) + 0.01])
            if obs.depth.size > 1:
                obs = obs.integrateInDepth(mean=True)
                shp = list(obs.data.shape)
                shp.insert(1, 1)
                obs.data.shape = shp
                obs.depth = np.asarray([self.keywords["depth"]])
                obs.depth_bnds = np.asarray(
                    [[float(self.keywords["depth"]) - 0.01, float(self.keywords["depth"]) + 0.01]]
                )
                obs.layered = True
                obs.name = self.variable

        # Try to extract a commensurate quantity from the model
        mod = m.extractTimeSeries(
            self.variable,
            alt_vars=self.alternate_vars,
            expression=self.derived,
            initial_time=obs.time_bnds[0, 0],
            final_time=obs.time_bnds[-1, 1],
            lats=None if obs.spatial else obs.lat,
            lons=None if obs.spatial else obs.lon,
        )
        obs, mod = il.MakeComparable(
            obs,
            mod,
            mask_ref=True,
            clip_ref=True,
            extents=self.extents,
            logstring="[%s][%s]" % (self.longname, m.name),
        )

        return obs, mod

    # def modelPlots(self, m):
    #     r""" here we define some types of oceanic plots. Before that
    #     # some of the plots can be generated using the standard
    #     # routine, with some modifications
    #     """

    #     super(ConfIOMB, self).modelPlots(m)

        #
        # bname = os.path.join(self.output_path, "%s_Benchmark.nc" % (self.name))
        # fname = os.path.join(self.output_path, "%s_%s.nc" % (self.name, m.name))

        # get the HTML page
        # page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        # print(os.path.join(self.output_path, "%s_global_%s.png" % (m.name, self.regions[0])))
        # print(bname,fname)

        # if not os.path.isfile(bname):
        #     return
        # if not os.path.isfile(fname):
        #     return


