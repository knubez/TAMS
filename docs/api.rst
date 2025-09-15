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

   tams.data.fetch_example
   tams.data.open_example
   tams.data.load_example
   tams.data.tb_from_ir
   tams.data.get_mergir
   tams.data.get_imerg


.. _example_datasets:

Example datasets
~~~~~~~~~~~~~~~~

These can be accessed with :func:`tams.data.open_example` or
:func:`tams.data.load_example`.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Relevant example notebooks
   * - ``msg-rad``
     - Satellite infrared radiance data.

       This comes from the EUMETSAT MSG SEVIRI instrument,
       specifically the 10.8 μm channel (ch9).

       This dataset contains 6 time steps of 2-hourly data (every 2 hours):
       2006-09-01 00--10
     -
   * - ``msg-tb``
     - As in ``msg-tb`` but with :func:`tams.data.tb_from_ir` applied
       to convert radiance to brightness temperature (K).
     - :doc:`/examples/sample-satellite-data`, :doc:`/examples/identify`
   * - ``imerg``
     - GPM IMERG precipitation data.

       This is left-labeled half-hourly precipitation rate (mm/hr)
       fetched using :func:`tams.data.get_imerg`
       for the same period as the MSG data (2006-09-01 00:00--11:30;
       extended so that we have data for the full last two-hour period).
     - :doc:`/examples/sample-satellite-data`
   * - ``mpas-regridded``
     - MPAS-A model output (regridded).

       This is a spatial and variable subset of native MPAS output,
       Furthermore, it has been regridded to a regular lat/lon grid (0.25°)
       from the original 15-km mesh.

       After regridding, it was spatially subsetted so that
       lat ranges from -5 to 40°N
       and lon from 85 to 170°E.
       This domain relates to the PRECIP field campaign.

       It has ``tb`` (estimated brightness temperature)
       and ``pr`` (precipitation rate, derived by summing the MPAS accumulated
       grid-scale and convective precip variables ``rainnc`` and ``rainc`` and differentiating).

       ``tb`` was estimated using the (black-body) Stefan--Boltzmann law:

       .. math::
          E = \sigma T^4
          \implies T = (E / \sigma)^{1/4}

       where :math:`E` is the OLR (outgoing longwave radiation, ``olrtoa`` in MPAS output)
       in W m\ :sup:`-2`
       and :math:`\sigma` is the Stefan--Boltzmann constant.

       This dataset contains 127 time steps of hourly data:
       2006-09-08 12 -- 2006-09-13 18.
     - :doc:`/examples/tams-run`, :doc:`/examples/tracking-options`,
       :doc:`/examples/sample-mpas-ug-data`
   * - ``mpas-native``
     - MPAS-A native (unstructured grid) model output.

       This is a spatial and variable subset of native 15-km global mesh MPAS output.

       It has been spatially subsetted so that
       lat ranges from -5 to 20°N
       and lon from 85 to 170°E,
       similar to the example regridded MPAS dataset (``mpas-regridded``),
       except for a smaller lat upper bound.

       Like the regridded MPAS dataset, it has hourly
       ``tb`` (estimated brightness temperature)
       and ``precip`` (precipitation rate)
       for the period
       2006-09-08 12 -- 2006-09-13 18.

       Like the regridded MPAS dataset (``mpas-regridded``),
       ``tb`` was estimated using the (black-body) Stefan--Boltzmann law:

       .. math::
          E = \sigma T^4
          \implies T = (E / \sigma)^{1/4}

       where :math:`E` is the OLR (outgoing longwave radiation, ``olrtoa`` in MPAS output)
       in W m\ :sup:`-2`
       and :math:`\sigma` is the Stefan--Boltzmann constant.
     - :doc:`/examples/sample-mpas-ug-data`
   * - ``mosa-test-1``, ..., ``mosa-test-4``
     - Small idealized/test datasets from the MOSA paper :cite:p:`Prein_etal_2024`.
     -

External data sources
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function
     - Description
     - Relevant example notebooks
   * - :func:`tams.data.get_mergir`
     - GPM MERGIR brightness temperature from NASA.
     - :doc:`/examples/get`
   * - :func:`tams.data.get_imerg`
     - GPM IMERG precipitation data from NASA.
     - :doc:`/examples/get`

Utilities
---------

.. autosummary::
   :toctree: api/

   tams.plot_tracked

Options
-------

.. autosummary::
   :toctree: api/

   tams.set_options
   tams.get_options
