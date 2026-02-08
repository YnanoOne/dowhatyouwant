# Author: Felipe Benavides (PipeB62)
# Created: 01.02.2026
# Description: This script defines the Propagator class for calculation of scalar wave propagation.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Propagator():
    """
    Calculation of free space propagation of scalar wave using exact transfer function aproach.
    """

    def __init__(self,wavelength):
        
        self.wavelength = wavelength

        self.input_field = None
        self.domain_size = None
        self.N_samples = None
        self.step_size = None

        self.angular_spectrum = None
        self.frequency_mesh = None

        self.zvalues = None
        self.output_field = None
        
    def load_input_field(self,
                         field,
                         domain_size):
        """
        Function to load the scalar field at z=0
        
        :param field:
        Array with sampled field values at z=0 

        :param domain_size: Tuple with (length_x,length_y) of the sampled window.
        """

        self.input_field = field
        self.domain_size = np.array(domain_size)[::-1] #Reverse order to have rows,cols instead of Lx,Ly 

        self.N_samples = self.input_field.shape
        self.step_size = np.divide(self.domain_size,self.N_samples)

    def get_angular_spectrum(self):
        """
        Calculation of angular spectrum using fft.
        """

        spectrum0 = np.fft.fft2(self.input_field)
        self.angular_spectrum = spectrum0

        #Extract frequencies in correct order
        frequency_x = np.fft.fftfreq(self.N_samples[1],d=self.step_size[1]).astype(np.complex64)
        frequency_y = np.fft.fftfreq(self.N_samples[0],d=self.step_size[0]).astype(np.complex64)

        frequency_mesh_x,frequency_mesh_y = np.meshgrid(frequency_x,frequency_y)
        self.frequency_mesh = [frequency_mesh_x,frequency_mesh_y]

    def propagate(self,zvalues):
        """
        Calculation of propagated field at given z values using exact transfer function approach.
        
        :param zvalues: Array with desired z values for calculation.
        """
        
        self.zvalues = np.array(zvalues)
        # N_z = zvalues.size

        frequency_z = np.sqrt((1/self.wavelength**2 - self.frequency_mesh[0]**2 - self.frequency_mesh[1]**2).astype(np.complex128)) 
        # frequency_z = np.tile(frequency_z[:,:,np.newaxis],(1,1,N_z)) # This takes some memory I think --YnanoOne
        frequency_z = frequency_z[:,:,np.newaxis]

        translated_spectrum = self.angular_spectrum[:,:,np.newaxis]*np.exp(2*np.pi*1j*frequency_z*self.zvalues[np.newaxis,np.newaxis,:])

        self.output_field = np.fft.ifft2(translated_spectrum,axes=(0,1))

    def plot_xy(self,
                z,
                xlim = None, 
                ylim = None):
        
        """
        Plot field intensity in the xy plane at a given z value.

        :param z: desired z value 
        :param xlim: tuple with x limits (xmin,xmax)
        :param ylim: tuple with y limits (ymin,ymax)
        """

        z_ix = np.argmin(np.abs(self.zvalues-z))
        selected_z = self.zvalues[z_ix]
        extent = (-self.domain_size[1]/2 - self.step_size[1]/2,
                  self.domain_size[1]/2 + self.step_size[1]/2,
                  -self.domain_size[0]/2 - self.step_size[0]/2,
                  self.domain_size[0]/2 + self.step_size[0]/2,)

        fig,ax = plt.subplots()
        g = ax.imshow(np.abs(self.output_field[:,:,z_ix]), 
                      origin = "lower",
                      extent = extent)
        fig.colorbar(g)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"z = {selected_z:.2f}")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig,ax

 
    def plot_zy(self,
                x,
                zlim = None,
                ylim = None):
        
        """
        Plot field intensity in the zy plane at a given x value.

        :param x: desired x value 
        :param zlim: tuple with z limits (zmin,zmax)
        :param ylim: tuple with y limits (ymin,ymax)
        """
        
        xs = np.linspace(-self.domain_size[1]/2,self.domain_size[1]/2,self.N_samples[1])

        x_ix = np.argmin(np.abs(xs - x))
        selected_x = xs[x_ix]

        zstep = np.abs(self.zvalues[1]-self.zvalues[0])
        extent = (self.zvalues[0] - zstep/2,
                  self.zvalues[-1] + zstep/2,
                  -self.domain_size[0]/2 - self.step_size[0]/2,
                  self.domain_size[0]/2 + self.step_size[0]/2,)

        fig,ax = plt.subplots()
        g = ax.imshow(np.abs(self.output_field[:,x_ix,:]), 
                      origin = "lower",
                      extent = extent)
        fig.colorbar(g)
        
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_title(f"x = {selected_x:.2f}")
        if zlim is not None:
            ax.set_xlim(zlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig,ax

    def animation_xy(self,
                     xlim=None,
                     ylim=None):
        """
        Create animation of field intensity in the xy plane at all calculated z values
        
        :param xlim: tuple with x limits (xmin,xmax)
        :param ylim: tuple with y limits (ymin,ymax)
        """

        output_intensity = np.abs(self.output_field)
        extent = (-self.domain_size[1]/2 - self.step_size[1]/2,
                  self.domain_size[1]/2 + self.step_size[1]/2,
                  -self.domain_size[0]/2 - self.step_size[0]/2,
                  self.domain_size[0]/2 + self.step_size[0]/2,)
        
        fig, ax = plt.subplots()

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in
        # each frame
        ims = []
        for i in range(self.output_field.shape[-1]):
            im = ax.imshow(output_intensity[:,:,i], origin="lower", extent=extent, animated=True)
            if i == 0:
                ax.imshow(output_intensity[:,:,i], origin="lower",extent=extent)  # show an initial one first
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)

        # To save the animation, use e.g.
        #
        # ani.save("movie.mp4")
        #
        # or
        #
        # writer = animation.FFMpegWriter(
        #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save("movie.mp4", writer=writer)

        return ani

    

    
    

