import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

def plotmap(lons,lats,variable,savefile=None, cmap="RdBu_r", levels=None,extend='both',
            variable_label='',plottitle='',plotaxis=None,colorbar=1,alpha=1, projection=None):
    """ Plots a map and displays output or saves to file with path 
    and filename savefile (string). Inputs are lons, lats and the variable
    in format they are outputted from ReadFile (ie from netcdf file). 
    also cmap is colour- eg for tempchange choose RdBu_r"""
    if plotaxis is None:
        plt.clf()
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        if projection is None:
            projection = ccrs.PlateCarree()
        ax = plt.axes(projection=projection)

    else:
        plt.sca(plotaxis)
        ax = plotaxis
    ax.coastlines()

    variable, lons = add_cyclic_point(variable, coord=lons)
    
    # Plot map
    conmap = ax.contourf(lons, lats, variable, cmap=cmap, 
                        levels=levels, extend=extend, alpha=alpha,
                        transform=ccrs.PlateCarree()  )
    if colorbar == 1.0:
        sm = plt.cm.ScalarMappable(norm=colors.Normalize(levels[0], levels[-1]), cmap=cmap)
        sm._A = []
        if levels[-1] <= 0.01:
            cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(levels[0], levels[-1], 0.001))
        elif levels[-1]<=0.1:
            cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(levels[0], levels[-1], 0.01))
        elif levels[-1] > 1.:
            cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(levels[0], levels[-1], 0.5))
        else:
            cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(levels[0], levels[-1], 0.1))
        cbar.set_label(variable_label)

    plt.title(plottitle)
    plt.tight_layout()

    if type(savefile) is str:
        plt.savefig(savefile)
        print('Saved plot as '+savefile)
    return ax, conmap

def add_hatching(lons, lats, hatching, ax):
    hatching, lons = add_cyclic_point(hatching, coord=lons)
    hatches=['', '..', '..', 'oo']
    ax.contourf(lons, lats, hatching, colors = None, levels = [-1. , 0., 1.0, 2.], hatches=hatches,
                 alpha = 0.)

def plotmap_uncertainty(lons, lats, variable, uncertainty, savefile=None, cmap="RdBu_r", levels=None,extend='both',
            variable_label='',plottitle='',plotaxis=None,colorbar=1,alpha=1):
    """ Plots a map and displays output or saves to file with path 
    and filename savefile (string). Inputs are lons, lats and the variable
    in format they are outputted from ReadFile (ie from netcdf file). 
    also cmap is colour- eg for tempchange choose RdBu_r"""
    if plotaxis is None:
        plt.clf()
        plt.figure(figsize=(12, 5))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    variable, _ = add_cyclic_point(variable, coord=lons)
    uncertainty, lons = add_cyclic_point(uncertainty, coord=lons) 
    """   
    # Create an alpha channel based on uncertainty values
    # Any value whose absolute value is > .1 will have zero transparency
    alphas = colors.Normalize(0, .3, clip=True)(uncertainty)
    alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4

    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
    vmax = np.abs(variable).max()
    vmin = -vmax
    cols = colors.Normalize(vmin, vmax)(variable)
    colormap = plt.get_cmap(cmap)
    cols = colormap(cols)

    # Now set the alpha channel to the one we created above
    cols[..., -1] = alphas
    """
    # Plot map
    conmap = ax.contourf(lons, lats, variable, cmap=cmap,
                        levels=levels, extend=extend)
    # Now add uncertainty
    grey = plt.get_cmap('Greys_r')
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'new', grey(np.linspace(0., 0.3, 10)))
    uncert_levels = np.arange(1.0, 2.51, 0.5)
    ax.contourf(lons, lats, uncertainty, cmap = new_cmap, levels = uncert_levels,
                 alpha = 0.2, extend='max')
  
    if colorbar == 1.0:
        sm = plt.cm.ScalarMappable(norm=colors.Normalize(levels[0], levels[-1]), cmap=cmap)
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(levels[0], levels[-1], 0.5))
        cbar.set_label(variable_label) 
    plt.xlabel('Longitude')
    plt.xticks(np.arange(-180.,185.,90.))
    plt.yticks(np.arange(-90.,91.,30.))
    plt.ylabel('Latitude')
    plt.title(plottitle)
    plt.tight_layout()

    if type(savefile) is str:
        plt.savefig(savefile)
        print('Saved plot as '+savefile)
    return ax, conmap


