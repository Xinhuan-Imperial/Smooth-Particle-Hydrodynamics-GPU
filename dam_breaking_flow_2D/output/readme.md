The output will be written in .csv format, and can be loaded using Paraview: https://www.paraview.org/. SPH is a meshless method, while 2D or 3D images are based on rectangular mesh. As a result, interpolation is necessary for visualization, follow this Paraview tutorial: https://blog.kitware.com/point-and-smoothed-particle-hydrodynamics-sph-interpolation-in-vtk/. Otherwise, it is simple to conduct interpolation to a rectangular grid for visualization.