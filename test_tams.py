import tams

r = tams.load_example_ir().isel(time=0)

tb = tams.tb_from_ir(r, ch=9)


def test_ch9_tb_loaded():
    assert tb.name == "ch9"
    assert tuple(tb.coords) == ("lon", "lat", "time")


def test_data_in_contours_methods_same_result():
    cs235 = tams._contours_to_gdf(tams.contours(tb, 235))
    cs219 = tams._contours_to_gdf(tams.contours(tb, 219))
    cs235, _ = tams._size_filter_contours(cs235, cs219)

    x1 = tams._data_in_contours_sjoin(tb, cs235)
    x2 = tams._data_in_contours_regionmask(tb, cs235)
    assert (x1.count_ch9 > 0).all()
    assert x1.count_ch9.equals(x2.count_ch9)
    assert x1.std_ch9.equals(x2.std_ch9)
    # TODO: mean values aren't exactly the same


def test_load_mpas_sample():
    ds = tams.load_example_mpas()
    assert tuple(ds.data_vars) == ("tb", "precip")
    assert tuple(ds.coords) == ("time", "lon", "lat")
