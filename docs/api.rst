===
API
===

Core
----

These functions make up the core of the TAMS algorithm:

.. autosummary::
   :toctree: api/

   tams.identify
   tams.track
   tams.classify


The helper function :func:`tams.run` combines the above plus additional processing,
including computing stats on gridded data within the identified cloud element regions.

.. autosummary::
   :toctree: api/

   tams.run


Lower level functions used in the above include:

.. autosummary::
   :toctree: api/

   tams.calc_ellipse_eccen
   tams.contours
   tams.data_in_contours
   tams.overlap
   tams.project


Data
----

.. autosummary::
   :toctree: api/

   tams.load_example_tb
   tams.load_example_mpas
   tams.load_example_mpas_ug
   tams.load_mpas_precip
   tams.data.download_examples
   tams.data.load_example_ir
   tams.data.tb_from_ir
   tams.data.get_mergir
   tams.data.get_imerg


Utilities
---------

.. autosummary::
   :toctree: api/

   tams.plot_tracked


Idealized
---------

.. automodule:: tams.idealized

.. autosummary::
   :toctree: api/

   Blob
   Field
   Sim

Blob properties:

.. autosummary::
   :toctree: api/

   Blob.center
   Blob.ring
   Blob.polygon

Transform blobs:

.. autosummary::
   :toctree: api/

   Blob.set_tendency
   Blob.get_tendency
   Blob.evolve
   Blob.merge
   Blob.split
   Blob.copy
   Blob.to_geopandas

Compute the blob's impact on the field:

.. autosummary::
   :toctree: api/

   Blob.well

Transform a collection of blobs:

.. autosummary::
   :toctree: api/

   Field.evolve
   Field.to_xarray
   Field.to_geopandas
   Field.copy

Simulate a collection of blobs:

.. autosummary::
   :toctree: api/

   Sim.advance
   Sim.to_xarray
