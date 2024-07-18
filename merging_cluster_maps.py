import numpy as np
import illustris_python as il
import pandas as pd
import h5py

cat = pd.read_csv('fiducial/dataset/m_test.csv')
basePath = '/virgotng/mpia/TNG-Cluster/TNG-Cluster/postprocessing/projections/'

images_halfr200c = h5py.File(basePath + 'gas-xray_lum_0.5-5.0kev_099_0.5r500_d=r200.hdf5')
images_2r200c = h5py.File(basePath + 'gas-xray_lum_0.5-5.0kev_099_2r200_d=r200.hdf5')
names = [key for key in images_halfr200c.keys() if 'Halo_' in key]

cat = cat[cat['snapshot_id'] == 99]
fdiff = np.zeros(len(cat))
i = 0
for ind in cat.index:
    hnum = cat['halo_num'][ind]
    axis = cat['projection'][ind] 
    im2rc = images_2r200c['Halo_%d' % hnum][:,:,axis]
    im05rc = images_halfr200c['Halo_%d' % hnum][:,:,axis]
    hind = names.index('Halo_%d' % hnum)
    pixel_size_05rc = 4*0.5*images_halfr200c['r500c'][hind]/2000
    pixel_size_2rc = 4*images_2r200c['r200c'][hind]/2000
    pixel_area05rc = pixel_size_05rc ** 2
    pixel_area2rc = pixel_size_2rc ** 2
    diff = im2rc.sum() * pixel_area2rc - (im05rc.sum() * pixel_area05rc)
    fdiff[i] = diff/(im2rc.sum() * pixel_area2rc)
    i += 1

cat.insert(10,'delta_emission_05_2_r200c', fdiff)
cat.to_csv('missed_emission.csv')

