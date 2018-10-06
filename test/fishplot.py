import numpy as np
import os as os
import sys as sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import copy
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter

FS=18

formatter=ScalarFormatter(useOffset=False)

def plot_fisher_single(params,name,covar,typs,ax,prop,fact_axis,fs=FS) :
    nb=128

    sigma_max=0
    title="$\\sigma($"+params[name]['label']+"$)=["
    for i in np.arange(len(covar)) :
        if typs[i]=='covar' :
            sigma=np.sqrt(covar[i][name][name])
            x_arr=params[name]['value']-4*sigma+8*sigma*np.arange(nb)/(nb-1.)
            p_arr=np.exp(-(x_arr-params[name]['value'])**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
            ax.plot(x_arr,p_arr,color=prop[i]['col'],linestyle=prop[i]['ls'],linewidth=prop[i]['lw'])
        else :
            sigma=np.std(covar[i][name])
            mean=np.mean(covar[i][name])
            h,b=np.histogram(covar[i][name],bins=40,range=[mean-4*sigma,mean+4*sigma])
            x_arr=b[:-1]
            p_arr=(h+0.)/(np.sum(h)*(b[1]-b[0]))
            ax.step(x_arr,p_arr,where='post',color=prop[i]['col'],linestyle=prop[i]['ls'],
                    linewidth=prop[i]['lw'])
        if sigma>=sigma_max :
            sigma_max=sigma
        title+="%.3lf"%sigma
        if i<(len(covar)-1) :
            title+=","
    title+="]$"
    ax.set_title(title)
    ax.set_xlim([params[name]['value']-fact_axis*sigma_max,params[name]['value']+fact_axis*sigma_max])
    ax.set_xlabel(params[name]['label'],fontsize=fs)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    for label in ax.get_yticklabels():
        label.set_fontsize(fs-6)
    for label in ax.get_xticklabels():
        label.set_fontsize(fs-6)

def plot_fisher_two(params,name1,name2,covar,typs,ax,prop,fact_axis,add_2sigma=False,fs=FS) :
    sig0_max=0
    sig1_max=0
    for i in np.arange(len(covar)) :
        if typs[i]=='covar' :
            cov=np.zeros([2,2])
            cov[0,0]=covar[i][name1][name1]
            cov[0,1]=covar[i][name1][name2]
            cov[1,0]=covar[i][name2][name1]
            cov[1,1]=covar[i][name2][name2]
            sig0=np.sqrt(cov[0,0])
            sig1=np.sqrt(cov[1,1])
            mean0=params[name1]['value']
            mean1=params[name2]['value']
            
            w,v=np.linalg.eigh(cov)
            angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
            a_1s=np.sqrt(2.3*w[0])
            b_1s=np.sqrt(2.3*w[1])
            a_2s=np.sqrt(6.17*w[0])
            b_2s=np.sqrt(6.17*w[1])

            centre=np.array([params[name1]['value'],params[name2]['value']])
            
            if prop[i]['alpha']<0 :
                fcol='none'
                ecol=prop[i]['col']
                alpha=1.
            else :
                fcol=prop[i]['col']
                ecol=prop[i]['col']
                alpha=prop[i]['alpha']

            e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                         facecolor=fcol,linewidth=prop[i]['lw'],
                         linestyle=prop[i]['ls'],edgecolor=ecol,alpha=alpha)
#        if add_2sigma :
#            e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
#                         facecolor=fc[i],linewidth=lw[i],linestyle='dashed',edgecolor=lc[i])
#
#            ax.add_artist(e_2s)
            ax.add_artist(e_1s)
        else :
            mean0=np.mean(covar[i][name1])
            mean1=np.mean(covar[i][name2])
            sig0=np.std(covar[i][name1])
            sig1=np.std(covar[i][name2])
            L,xbins,ybins=np.histogram2d(covar[i][name1],covar[i][name2],
                                         range=[[mean0-4*sig0,mean0+4*sig0],[mean1-4*sig1,mean1+4*sig1]],
                                         bins=30)
            shape=L.shape
            L=L.ravel()
            i_sort=np.argsort(L)[::-1]
            i_unsort=np.argsort(i_sort)
            L_cumsum=L[i_sort].cumsum()
            L_cumsum/=L_cumsum[-1]
            xbins=0.5*(xbins[1:]+xbins[:-1])
            ybins=0.5*(ybins[1:]+ybins[:-1])
            p=L_cumsum[i_unsort].reshape(shape)
            ax.contour(xbins,ybins,p.T,levels=[0.683,0.955],colors=prop[i]['col'],linewidths=prop[i]['lw'],
                       linestyles=['solid','dashed'])

        if sig0>=sig0_max :
            sig0_max=sig0
        if sig1>=sig1_max :
            sig1_max=sig1

        ax.set_xlim([mean0-fact_axis*sig0_max,mean0+fact_axis*sig0_max])
        ax.set_ylim([mean1-fact_axis*sig1_max,mean1+fact_axis*sig1_max])
        ax.set_xlabel(params[name1]['label'],fontsize=fs)
        ax.set_ylabel(params[name2]['label'],fontsize=fs)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    for label in ax.get_yticklabels():
        label.set_fontsize(fs-6)
    for label in ax.get_xticklabels():
        label.set_fontsize(fs-6)


def plot_fisher_all(params, #Parameters in the FMs
                    covar, #Covars to plot
                    typs, #Is covar or chains?
                    prop,#fc,lw,ls,lc, #Foreground colors, line widths, line styles and line colours for each FM
                    labels, #Labels for each FM
                    fact_axis, #The x and y axes will be fact_axis x error in each parameter
                    fname,
                    do_1D=True,
                    do_titles=True,
                    fs=FS,
                    nticks=4) : #File to save the plot
    n_params=len(params)

    fig=plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0,wspace=0)
    for i,name1 in enumerate(params) :
        i_col=i
        for j,name2 in enumerate(params) :#in np.arange(n_params-i)+i :
            if j<i :
                continue
            i_row=j
            if do_1D :
                iplot=i_col+n_params*i_row+1
            else :
                iplot=i_col+(n_params-1)*(i_row-1)+1

            ax=None
            if do_1D :
                ax=fig.add_subplot(n_params,n_params,iplot)
                if i==j :
                    plot_fisher_single(params,name1,covar,typs,
                                       ax,prop,fact_axis,fs=fs)
            if i!=j :
                if do_1D :
                    ax=fig.add_subplot(n_params,n_params,iplot)
                else :
                    ax=fig.add_subplot(n_params-1,n_params-1,iplot)
                plot_fisher_two(params,name1,name2,covar,typs,ax,prop,fact_axis,fs=fs)
                if do_1D==False :
                    if n_params==2 :
                        leg_items=[]
                        for i in np.arange(len(covar)) :
                            leg_items.append(plt.Line2D((0,1),(0,0),color=prop[i]['col'],
                                                        linestyle=prop[i]['ls'],
                                                        linewidth=prop[i]['lw']))
                        ax.legend(leg_items,labels[:len(covar)],loc='upper left',frameon=False,fontsize=FS,ncol=2) 

            if ax!=None :
                if i_row!=n_params-1 :
                    ax.get_xaxis().set_visible(False)
                else :
                    plt.setp(ax.get_xticklabels(),rotation=45)

                if i_col!=0 :
                    ax.get_yaxis().set_visible(False)

                if i_col==0 and i_row==0 :
                    ax.get_yaxis().set_visible(False)
                
                ax.locator_params(nbins=nticks)

    if n_params!=2 :
        if n_params>1 : #Add labels in a separate plot
            if do_1D :
                ax=fig.add_subplot(n_params,n_params,2)
            else :
                ax=fig.add_subplot(n_params-1,n_params-1,2)
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            for i in np.arange(len(covar)) :
                ax.plot([-1,1],[-3,-3],color=prop[i]['col'],linestyle=prop[i]['ls'],
                        linewidth=2,label=labels[i])
            ax.legend(loc='upper left',frameon=False,fontsize=FS)
            ax.axis('off')
        else :
            ax.legend(loc='upper right',frameon=False,fontsize=FS) 
#        ax.axis('off')

    if fname!="none" :
        plt.savefig(fname,bbox_inches='tight')

#    plt.show()
