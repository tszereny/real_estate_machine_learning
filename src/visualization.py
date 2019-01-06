import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from src.utils import calc_intervals

def calc_fig_size(pic_ratio, multiplier=2):
    width=10*multiplier
    height=pic_ratio*multiplier*10
    return (width, height)    

def generate_colors(df, c_map, condition):
    c=[]
    for k, s in df.items():
        c_ord=[c_map[k][0]]*s[s==condition].count() + [c_map[k][1]]*s[s!=condition].count()
        c.append(c_ord)
    return np.array(c)    
    
def plot_na_ratio(df, grouping, subplot_n=2,  dest_dir='.'):
    title='Missing values-ratio by features (%)'
    na_count=df.groupby(grouping).count()
    total=df.groupby(grouping).agg(lambda col: len(col))
    na_total_ratio=100-na_count/total*100
    groups=na_total_ratio.index.tolist()
    sorted_df=na_total_ratio.T.sort_values(groups, ascending=True)
    col_names=sorted_df.index.tolist()
    ints=calc_intervals(subplot_n, len(col_names))
    fig, axs = plt.subplots(subplot_n)
    fig.suptitle(title, fontsize=20, weight=10, y=0.93)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

    for i, ax in enumerate(axs):
        ax_df=sorted_df.iloc[ints[i].start:ints[i].stop, :]
        ax_df.plot(kind='bar', figsize=(17,15), legend=True, ax=axs[i])
        axs[i].set_xticklabels(ax_df.index.tolist(), fontsize=14)
    fig.savefig('{0}/{1}.png'.format(dest_dir, title))
    return fig, axs
    
    
def plot_scatter_matrix(df, fs=15):
    col_n=len(df.columns)
    fig, axs = plt.subplots(col_n, col_n, figsize=(20,20))
    for c, c_k in enumerate(df):
        for r, r_k in enumerate(df):
            if c==r:
                axs[r,c].text(0.5, 0.5, s=c_k, horizontalalignment='center', verticalalignment='center', fontdict={'fontsize': fs})
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].yaxis.set_visible(False)
            elif (c==0) & (r==col_n-1):
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(True)
                axs[r,c].yaxis.set_visible(True)
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})
            elif c==0:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].yaxis.set_visible(True)
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})
            elif r==col_n-1:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(True)
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].yaxis.set_visible(False)
            elif (r==0) & (c==col_n-1):
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_ticks_position('top')
                axs[r,c].yaxis.set_ticks_position('right')
                axs[r,c].xaxis.set_label_position('top')
                axs[r,c].yaxis.set_label_position('right')
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})          
            elif c==col_n-1:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})
                axs[r,c].yaxis.set_ticks_position('right')
                axs[r,c].yaxis.set_label_position('right')
            elif r==0:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(True)
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].yaxis.set_visible(False)
                axs[r,c].xaxis.set_ticks_position('top')
                axs[r,c].xaxis.set_label_position('top')
            else:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].yaxis.set_visible(False)
            plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axs

def plot_scatter_map(figsize, x, y, c=None, colormap=None, cbar=True, c_bar_shrink=1, xlabel=None, ylabel=None, clabel=None, s=25, linewidth=0.25, alpha=1, epsg_code=4326, dpi=160):
    m = Basemap(urcrnrlat=y.max(),     # top
              urcrnrlon=x.max(),   # bottom
              llcrnrlat=y.min(),     # left
              llcrnrlon=x.min(),   # right
              epsg=epsg_code)
    width = figsize[0]
    height = figsize[1]
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    dpi = dpi
    xpixels = dpi * width
    m.arcgisimage(service='Canvas/World_Light_Gray_Base', xpixels=xpixels)
    plt.scatter(x=x, y=y, c=c, cmap=colormap, s=s, alpha=alpha, linewidth=linewidth, edgecolor='Black')
    plt.xticks(np.linspace(start=x.min(), stop=x.max(), num=np.ceil(width/2).astype(int)).round(2), fontsize=15)
    plt.yticks(np.linspace(start=y.min(), stop=y.max(), num=np.ceil(height/2).astype(int)).round(2), fontsize=15)
    plt.xlabel(x.name if xlabel is None else xlabel, fontsize=20)
    plt.ylabel(y.name if ylabel is None else ylabel, fontsize=20)
    if c is not None and cbar:
        cbar = plt.colorbar(orientation='vertical', shrink=c_bar_shrink)
        cbar.set_label(c.name if clabel is None else clabel, rotation=90, fontsize=20)
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15)
    return plt.gcf(), plt.gca()
    
def plot_sca_hist(df, x, y, bins, xlabel=None, ylabel=None, fs=15):
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(20,20)
    df.plot(kind='scatter', x=x, y=y, ax=axs[1,0], s=25, alpha=0.4, linewidth=0.25, edgecolor='Black', fontsize=fs)
    df.hist(x, ax=axs[0,0], bins=bins, linewidth=0.5, edgecolor='Black', grid=False, sharey=True, xlabelsize=fs, ylabelsize=fs, label=None)
    df.hist(y, ax=axs[1,1], bins=bins, linewidth=0.5, edgecolor='Black', grid=False, orientation='horizontal', sharex=True, xlabelsize=fs, ylabelsize=fs)
    axs[0,1].axis('off')
    axs[0,0].xaxis.set_ticks([])
    axs[1,1].yaxis.set_ticks([])
    axs[1,0].set_xlabel(x if xlabel is None else xlabel, fontsize=20)
    axs[1,0].set_ylabel(y if ylabel is None else ylabel, fontsize=20)
    axs[0,0].set_title('')
    axs[1,1].set_title('')
    plt.tight_layout(w_pad=-2, h_pad=-1.5)
    return fig, axs

def plot_sca_stackedhist(df, x, y, category, colors):
    FACE_C='lightgrey'
    df_by_categories=[df.loc[df[category]==cat_cond, [x, y, category]] for cat_cond in df[category].drop_duplicates()]
    x_by_cats=[df[x].values for df in df_by_categories]
    y_by_cats=[df[y].values for df in df_by_categories]
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(20,20)
    axs[1,0].set_facecolor(FACE_C)
    axs[1,0].grid(True)
    for x, y, c in zip(x_by_cats, y_by_cats, colors):
        axs[1,0].scatter(x=x, y=y, color=c, linewidth=0.25, edgecolor='Black', alpha=0.5)
    axs[0,0].set_facecolor(FACE_C)
    axs[0,0].grid(True)
    axs[0,0].hist(x_by_cats, color=colors, histtype='barstacked', bins=80, linewidth=0.3, edgecolor='Black')
    axs[1,1].set_facecolor(FACE_C)
    axs[1,1].grid(True)
    axs[1,1].hist(y_by_cats, color=colors, histtype='barstacked', bins=80, orientation='horizontal', linewidth=0.3, edgecolor='Black')
    axs[0,1].axis('off')
    plt.tight_layout()
    return fig, axs    
    
def plot_outliers(st_df, n_std=6, figsize=(30,10)):
    n_cols=len(st_df.columns)
    fig, axs = plt.subplots(1, n_cols, figsize=figsize)
    for ax, k in zip(axs, st_df):
        ax.scatter(range(len(st_df[k])), st_df[k], marker='o', s=10)
        ax.set_title(k, fontdict={'fontsize':20})
        for std in range(-n_std, n_std+1):
            ax.axhline(std, c='r')
    return fig, axs

def create_grid(x_min, x_max, y_min, y_max, data_points, model, x_first_feature=True):
    X, Y=np.meshgrid(np.linspace(x_min, x_max, data_points),
                         np.linspace(y_min, y_max, data_points))
    Z=np.ones((data_points,data_points))
    for i in range(0, data_points):
        if x_first_feature:
            Z[i] = model.predict(X=np.stack([X[i], Y[i]], axis=1))
        else:
            Z[i] = model.predict(X=np.stack([Y[i], X[i]], axis=1))
    return X, Y, Z

def plot_3d_surface(X, Y, Z, elevation=50, rotation=45, figsize=(15,15), saving=False, dir_name=None, model_name=None):
    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X=X, Y=Y, Z=Z, cmap="coolwarm")
    ax.view_init(elevation, rotation)
    if saving and model_name and dir_name:
        fn='model_{0}_elev_{1}_rotat_{2}.png'.format(model_name, elevation, rotation)
        fig.savefig(os.path.join(dir_name, fn))
    return fig, ax