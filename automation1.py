import os
import srtm
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import mapping, box
from matplotlib.colors import Normalize
from tqdm import tqdm
import shutil
import re
import logging
from whitebox.whitebox_tools import WhiteboxTools
import geopandas as gpd
import pandas as pd
import requests
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_city_visualizations(city_name, resolution=100):
    output_dir = os.path.abspath("output")
    os.makedirs(output_dir, exist_ok=True)

    # Adjust resolution for cities with potential data sparsity (e.g., Mumbai)
    if "Mumbai" in city_name:
        resolution = min(resolution, 50)  # Reduce resolution for Mumbai

    # Geocode city with error handling
    try:
        city_gdf = ox.geocode_to_gdf(city_name)
    except Exception as e:
        print(f"[❌] Geocoding failed for {city_name}: {e}")
        return []
    minx, miny, maxx, maxy = city_gdf.total_bounds

    # Basemap figure
    fig_basemap, ax = plt.subplots(figsize=(10, 6))
    city_gdf.to_crs(epsg=3857).plot(ax=ax, alpha=0.2, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title(f"Basemap of {city_name}")

    # Fetch elevation data
    try:
        latitudes = np.linspace(miny, maxy, resolution)
        longitudes = np.linspace(minx, maxx, resolution)
        elevation_data = srtm.get_data()
        elev_map = np.array([
            [elevation_data.get_elevation(lat, lon) or np.nan for lon in longitudes]
            for lat in latitudes
        ])
        if np.all(np.isnan(elev_map)):
            print("[❌] No valid elevation data retrieved.")
            return [fig_basemap]
        # Replace NaN with a small positive value to avoid issues
        elev_map = np.where(np.isnan(elev_map), 1.0, elev_map)
    except Exception as e:
        print(f"[❌] Elevation data retrieval failed: {e}")
        return [fig_basemap]

    # DEM figure
    fig_dem, ax_dem = plt.subplots(figsize=(10, 6))
    im = ax_dem.imshow(elev_map, cmap='terrain', extent=(minx, maxx, miny, maxy), origin='lower')
    plt.colorbar(im, ax=ax_dem, label="Elevation (m)")
    ax_dem.set_title(f"DEM of {city_name} from SRTM")
    ax_dem.set_xlabel("Longitude")
    ax_dem.set_ylabel("Latitude")

    # Save DEM as GeoTIFF with nodata value
    dem_path = os.path.join(output_dir, "dem.tif")
    transform = from_bounds(minx, miny, maxx, maxy, elev_map.shape[1], elev_map.shape[0])
    elev_map = elev_map.astype(np.float32)
    with rasterio.open(
        dem_path, "w",
        driver="GTiff",
        height=elev_map.shape[0],
        width=elev_map.shape[1],
        count=1,
        dtype=elev_map.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(elev_map, 1)

    # Terrain analysis
    wbt = WhiteboxTools()
    wbt.work_dir = output_dir
    wbt.verbose = False
    slope_path = os.path.join(output_dir, "slope.tif")
    aspect_path = os.path.join(output_dir, "aspect.tif")
    rugged_path = os.path.join(output_dir, "ruggedness.tif")
    tpi_path = os.path.join(output_dir, "tpi.tif")
    try:
        wbt.slope(dem=dem_path, output=slope_path)
        wbt.aspect(dem=dem_path, output=aspect_path)
        wbt.ruggedness_index(dem=dem_path, output=rugged_path)
        wbt.relative_topographic_position(
            dem=dem_path,
            output=tpi_path,
            filterx=9,
            filtery=9
        )
    except Exception as e:
        print(f"[❌] Terrain analysis failed: {e}")
        return [fig_basemap, fig_dem]

    # Function to visualize GeoTIFFs with specific colormaps and units
    def geotiff_to_figure(path, title, cmap='terrain', label="Value"):
        if not os.path.exists(path):
            print(f"[❌] File not found: {path}")
            return None
        with rasterio.open(path) as src:
            data = src.read(1, masked=True)
            # Replace invalid values with 0
            data = np.where(~np.isfinite(data), 0, data)
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(data, cmap=cmap, extent=src.bounds, origin='lower')
            plt.colorbar(im, ax=ax, label=label)
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            return fig

    # Create figures for terrain analysis outputs
    figs = [fig_basemap, fig_dem]
    for path, title, cmap, label in [
        (slope_path, "Slope Map", "viridis", "Degrees"),
        (aspect_path, "Aspect Map", "hsv", "Degrees"),
        (rugged_path, "Ruggedness Index Map", "terrain", "Unitless"),
        (tpi_path, "Topographic Position Index (TPI)", "terrain", "Unitless")
    ]:
        fig = geotiff_to_figure(path, title, cmap, label)
        if fig:
            figs.append(fig)

    # Save figures to output directory
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(output_dir, f"figure_{i}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return figs

def get_and_save_lulc(city_name, output_dir="output"):
    print(f"[INFO] Fetching LULC data for {city_name}...")
    tags = {
        "landuse": True,
        "natural": ["water", "wood", "scrub", "wetland", "grassland"],
        "leisure": ["park", "garden", "golf_course"]
    }
    try:
        gdf = ox.features_from_place(city_name, tags=tags)
    except Exception as e:
        print(f"[❌] Failed to fetch data for {city_name}: {e}")
        return None
    if gdf.empty:
        print(f"[❌] No LULC features found for {city_name}.")
        return None
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    def get_lulc_category(row):
        if pd.notna(row.get("landuse")):
            return f"landuse_{row['landuse']}"
        elif pd.notna(row.get("natural")):
            return f"natural_{row['natural']}"
        elif pd.notna(row.get("leisure")):
            return f"leisure_{row['leisure']}"
        return "unknown"
    gdf["lulc_category"] = gdf.apply(get_lulc_category, axis=1)
    print(f"[✅] Retrieved {len(gdf)} features for LULC.")
    try:
        os.makedirs(output_dir, exist_ok=True)
        city_safe = city_name.replace(' ', '_')
        lulc_path = os.path.join(output_dir, f"lulc_{city_safe}.geojson")
        gdf.to_file(lulc_path, driver="GeoJSON")
        print(f"[✅] Saved LULC data to: {lulc_path}")
    except Exception as e:
        print(f"[❌] Failed to save GeoJSON: {e}")
        return gdf
    return gdf

def plot_lulc(gdf, city_name, output_dir="output"):
    print(f"[INFO] Plotting LULC map for {city_name}...")
    if gdf is None or gdf.empty:
        print(f"[❌] No data to plot for {city_name}.")
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        gdf.to_crs(epsg=3857).plot(
            column="lulc_category",
            cmap="Set3",
            legend=True,
            ax=ax,
            edgecolor="k",
            alpha=0.7,
            legend_kwds={"loc": "center left", "bbox_to_anchor": (1, 0.5), "frameon": False}
        )
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title(f"Land Use / Land Cover Map of {city_name}")
        plt.axis("off")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "lulc_map.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"[✅] Saved LULC map to: {plot_path}")
        plt.show()
    except Exception as e:
        print(f"[❌] Failed to plot LULC map: {e}")
    finally:
        plt.close(fig)

def fetch_and_plot_hydrology(city_name, output_folder="output"):
    print(f"[INFO] Fetching hydrological features for {city_name}...")
    results = {"data": None, "figure": None}
    os.makedirs(output_folder, exist_ok=True)
    tags = {
        "waterway": ["river", "stream", "canal", "drain", "ditch"],
        "natural": ["water", "wetland", "coastline"],
        "landuse": ["reservoir", "basin"],
        "water": ["lake", "river", "pond", "canal"]
    }
    try:
        gdf = ox.features_from_place(city_name, tags=tags)
    except Exception as e:
        print(f"[❌] Failed to fetch data for {city_name}: {e}")
        return results
    if gdf.empty:
        print(f"[❌] No hydrological features found for {city_name}.")
        return results
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])]
    def get_hydro_category(row):
        if pd.notna(row.get("waterway")):
            return f"waterway_{row['waterway']}"
        elif pd.notna(row.get("natural")):
            return f"natural_{row['natural']}"
        elif pd.notna(row.get("landuse")):
            return f"landuse_{row['landuse']}"
        elif pd.notna(row.get("water")):
            return f"water_{row['water']}"
        return "unknown"
    gdf["hydro_category"] = gdf.apply(get_hydro_category, axis=1)
    print(f"[✅] Retrieved {len(gdf)} hydrological features.")
    out_path = os.path.join(output_folder, "hydro.geojson")
    try:
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        gdf.to_file(out_path, driver="GeoJSON")
        results["data"] = out_path
        print(f"[✅] Saved hydrological features to {out_path}")
    except Exception as e:
        print(f"[❌] Failed to save GeoJSON: {e}")
    if not gdf.empty:
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            for geom_type in ["LineString", "MultiLineString"]:
                subset = gdf[gdf.geometry.type == geom_type]
                if not subset.empty:
                    subset.to_crs(epsg=3857).plot(
                        column="hydro_category", cmap="Blues", ax=ax, alpha=0.7, linewidth=2
                    )
            for geom_type in ["Polygon", "MultiPolygon"]:
                subset = gdf[gdf.geometry.type == geom_type]
                if not subset.empty:
                    subset.to_crs(epsg=3857).plot(
                        column="hydro_category", cmap="Blues", ax=ax, alpha=0.7, edgecolor="black"
                    )
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
            ax.set_title(f"Hydrological Features in {city_name}")
            plt.axis("off")
            handles, labels = [], []
            for cat in gdf["hydro_category"].unique():
                handles.append(plt.Line2D([0], [0], color="blue", lw=2 if "waterway" in cat else 0,
                                        marker="s" if "waterway" not in cat else None, markersize=10))
                labels.append(cat.replace("_", ": "))
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
            plot_path = os.path.join(output_folder, "hydro_map.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            results["figure"] = plot_path
            print(f"[✅] Saved hydrological map to {plot_path}")
            plt.show()
        except Exception as e:
            print(f"[❌] Failed to plot hydrological map: {e}")
        finally:
            plt.close(fig)
    return results

def get_city_bbox(city_name):
    try:
        city_gdf = ox.geocode_to_gdf(city_name, which_result=1)
        bounds = city_gdf.total_bounds
        return bounds, city_gdf
    except Exception as e:
        print(f"[❌] Geocoding failed for {city_name}: {e}")
        raise

def verify_tiff_file(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            print(f"[INFO] TIFF file {tif_path} is valid. Shape: {src.shape}, Bands: {src.count}")
            return True
    except rasterio.errors.RasterioIOError as e:
        print(f"[ERROR] TIFF file {tif_path} is corrupted or invalid: {str(e)}")
        return False

def download_worldpop_raster(output_tif="city_population.tif"):
    url = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020.tif"
    if os.path.exists(output_tif):
        if verify_tiff_file(output_tif):
            print("[INFO] WorldPop raster already exists and is valid.")
            return output_tif
        else:
            os.remove(output_tif)
            print(f"[INFO] Removed corrupted file {output_tif}. Redownloading...")
    print("[INFO] Downloading WorldPop raster (~100m resolution for India)...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024
        with open(output_tif, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="Downloading", ncols=80
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        if verify_tiff_file(output_tif):
            print(f"[✅] Download complete: {output_tif}")
            return output_tif
        else:
            raise Exception(f"[ERROR] Downloaded file {output_tif} is invalid.")
    except requests.exceptions.RequestException as e:
        print(f"[❌] Download failed: {str(e)}. Check URL or network.")
        if os.path.exists(output_tif):
            os.remove(output_tif)
        raise

def clip_and_plot_population(city_gdf, raster_path, city_name, save_clipped=True):
    print("[INFO] Clipping population raster to city boundary...")
    try:
        with rasterio.open(raster_path) as src:
            bbox_size = max(city_gdf.total_bounds[2] - city_gdf.total_bounds[0],
                          city_gdf.total_bounds[3] - city_gdf.total_bounds[1])
            tolerance = max(0.0001, min(0.01, bbox_size / 1000))
            city_gdf_simplified = city_gdf.simplify(tolerance, preserve_topology=True)
            city_geom = [mapping(geom) for geom in city_gdf_simplified.geometry]
            if src.crs != city_gdf.crs:
                print("[WARNING] Raster and city CRS differ. Reprojecting city geometry.")
                city_gdf_simplified = city_gdf_simplified.to_crs(src.crs)
                city_geom = [mapping(geom) for geom in city_gdf_simplified.geometry]
            out_image, out_transform = mask(src, city_geom, crop=True, nodata=src.nodata)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": src.nodata
            })
        population = out_image[0]
        print(f"[DEBUG] Population stats — min: {population.min()}, max: {population.max()}")
        city_safe = city_name.replace(' ', '_')
        clipped_path = os.path.join("output", f"population_clipped_{city_safe}.tif")
        if save_clipped:
            with rasterio.open(clipped_path, "w", **out_meta) as dst:
                dst.write(out_image)
            print(f"[✅] Clipped raster saved: {clipped_path}")
        masked_pop = np.ma.masked_where(population == src.nodata, population)
        plt.figure(figsize=(10, 8))
        norm = Normalize(vmin=0, vmax=np.percentile(masked_pop[masked_pop > 0], 95))
        show(masked_pop, cmap="OrRd", norm=norm)
        plt.title(f"Population Density in {city_name} (WorldPop 2020)")
        plt.axis("off")
        plot_path = os.path.join("output", f"{city_safe}_population.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"[✅] Population plot saved: {plot_path}")
        plt.show()
    except rasterio.errors.RasterioIOError as e:
        print(f"[ERROR] Failed to clip raster: {str(e)}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error during clipping: {str(e)}")
        raise

def get_building_footprints(city_name="Pune, India"):
    logger.info(f"[INFO] Downloading building footprints for {city_name}...")
    tags = {"building": True}
    try:
        buildings = ox.features_from_place(city_name, tags=tags, which_result=1)
    except Exception as e:
        logger.error(f"[❌] Failed to fetch data for {city_name}: {e}")
        raise
    if buildings.empty:
        logger.warning(f"[⚠️] No building footprints found for {city_name}.")
        return gpd.GeoDataFrame()
    buildings = buildings[buildings.geometry.notnull()]
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    buildings.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', col) for col in buildings.columns]
    logger.info(f"[✅] Retrieved {len(buildings)} building footprints.")
    save_path = os.path.join(OUTPUT_DIR, "buildings.gpkg")
    if os.path.exists(save_path):
        logger.warning(f"[⚠️] {save_path} exists. Overwriting...")
    try:
        buildings.to_file(save_path, driver="GPKG")
        logger.info(f"[💾] Saved to: {save_path}")
    except Exception as e:
        logger.warning(f"[⚠️] Save failed due to complex fields: {e}")
        logger.info("[🔁] Retrying with only geometry column...")
        buildings_geometry = buildings[["geometry"]]
        buildings_geometry.to_file(save_path, driver="GPKG")
        logger.info(f"[💾] Saved geometry-only to: {save_path}")
    return buildings

def plot_buildings(buildings, title="Building Footprints", save_plot=True):
    logger.info("[🖼️] Plotting building footprints...")
    if buildings.empty:
        logger.warning("[⚠️] No buildings to plot.")
        return
    try:
        if len(buildings) > 5000:
            buildings = buildings.sample(frac=0.5, random_state=42)
            logger.info(f"[ℹ️] Sampled {len(buildings)} buildings for performance.")
        fig, ax = plt.subplots(figsize=(10, 10))
        buildings.to_crs(epsg=3857).plot(
            ax=ax, color="gray", edgecolor="black", linewidth=0.2, alpha=0.5
        )
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title(title)
        plt.axis("off")
        if save_plot:
            safe_title = re.sub(r'[^0-9a-zA-Z_]+', '_', title)
            plot_path = os.path.join(OUTPUT_DIR, f"{safe_title}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"[💾] Saved plot to: {plot_path}")
        plt.show()
    except Exception as e:
        logger.error(f"[❌] Failed to plot: {e}")
    finally:
        plt.close(fig)

def calculate_flow_accumulation(dem_path, output_dir="output"):
    print("[INFO] Starting D8 Flow Accumulation...")
    os.makedirs(output_dir, exist_ok=True)
    dem_path = os.path.abspath(dem_path)
    flow_accum_path = os.path.abspath(os.path.join(output_dir, "flow_accumulation.tif"))
    if not os.path.exists(dem_path):
        print(f"[❌] DEM file not found at: {dem_path}")
        return
    wbt = WhiteboxTools()
    wbt.work_dir = os.path.dirname(dem_path)
    wbt.verbose = True
    try:
        wbt.run_tool(
            "D8FlowAccumulation",
            [f"--input={dem_path}", f"--output={flow_accum_path}", "--out_type=cells"],
            callback=None
        )
        if os.path.exists(flow_accum_path):
            print(f"[✅] Flow accumulation map saved to: {flow_accum_path}")
        else:
            print("[❌] Output file was not created.")
    except Exception as e:
        print(f"[❌] Flow accumulation failed: {e}")
    try:
        with rasterio.open(flow_accum_path) as src:
            print(f"[📊] Output shape: {src.shape}, CRS: {src.crs}")
    except Exception as e:
        print(f"[⚠️] Could not read output GeoTIFF: {e}")

def fill_depressions(input_dem="output/dem.tif", output_dir="output"):
    print("[INFO] Filling depressions in DEM using WhiteboxTools...")
    input_path = os.path.abspath(input_dem)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dem_filled.tif")
    if not os.path.exists(input_path):
        print(f"[❌] DEM not found: {input_path}")
        return
    wbt = WhiteboxTools()
    wbt.work_dir = output_dir
    wbt.verbose = True
    try:
        wbt.run_tool("FillDepressions", [f"--dem={input_path}", f"--output={output_path}"])
        if os.path.exists(output_path):
            print(f"[✅] Filled DEM saved to: {output_path}")
        else:
            print("[❌] WhiteboxTools ran but the output file was not created.")
    except Exception as e:
        print(f"[❌] WhiteboxTools failed: {e}")
    return output_path

def calculate_flow_accumulation_filled(input_filled_dem="output/dem_filled.tif", output_dir="output"):
    print("[INFO] Running D8 Flow Accumulation on filled DEM...")
    input_path = os.path.abspath(input_filled_dem)
    output_dir = os.path.abspath(output_dir)
    output_path = os.path.join(output_dir, "flow_accumulation_filled.tif")
    os.makedirs(output_dir, exist_ok=True)
    wbt = WhiteboxTools()
    wbt.work_dir = output_dir
    wbt.verbose = True
    try:
        wbt.run_tool(
            "D8FlowAccumulation",
            [f"--input={input_path}", f"--output={output_path}", "--out_type=cells"]
        )
        if os.path.exists(output_path):
            print(f"[✅] Flow accumulation (filled DEM) saved to: {output_path}")
        else:
            print(f"[❌] Tool ran but file not created: {output_path}")
    except Exception as e:
        print(f"[❌] Flow accumulation failed: {e}")

def fix_and_generate_wetness_index(dem_path, output_dir="output"):
    dem_path = os.path.abspath(dem_path)
    output_dir = os.path.abspath(output_dir)
    dem_breached = os.path.join(output_dir, "dem_breached.tif")
    sca_path = os.path.join(output_dir, "flow_accum_breached.tif")
    wetness_path = os.path.join(output_dir, "wetness_index.tif")
    wbt = WhiteboxTools()
    wbt.work_dir = output_dir
    wbt.verbose = True
    print("[🧹] Step 1: Breaching depressions...")
    wbt.run_tool("BreachDepressions", [f"--dem={dem_path}", f"--output={dem_breached}"])
    print("[💧] Step 2: Flow Accumulation (D8)...")
    wbt.run_tool("D8FlowAccumulation", [
        f"--input={dem_breached}",
        f"--output={sca_path}",
        "--out_type=catchment area",
        "--log=false"
    ])
    print("[🌊] Step 3: Wetness Index...")
    wbt.run_tool("WetnessIndex", [f"--dem={dem_breached}", f"--sca={sca_path}", f"--output={wetness_path}"])
    if os.path.exists(wetness_path):
        print(f"[✅] Wetness Index saved: {wetness_path}")
    else:
        print("[❌] Wetness Index generation failed. Please inspect the input rasters.")

def fix_sca_raster(sca_path, fixed_output_path):
    with rasterio.open(sca_path) as src:
        profile = src.profile
        data = src.read(1)
        data = np.where(data <= 0, 0.0001, data)
    with rasterio.open(fixed_output_path, 'w', **profile) as dst:
        dst.write(data, 1)
    print(f"[🛠️] Fixed SCA raster saved to: {fixed_output_path}")

def rerun_wetness_index_fixed(dem_path, fixed_sca_path, output_path):
    wbt = WhiteboxTools()
    wbt.work_dir = os.path.dirname(output_path)
    wbt.verbose = True
    try:
        wbt.run_tool("WetnessIndex", [
            f"--dem={os.path.abspath(dem_path)}",
            f"--sca={os.path.abspath(fixed_sca_path)}",
            f"--output={os.path.abspath(output_path)}"
        ])
        print(f"[✅] Wetness Index created at: {output_path}")
    except Exception as e:
        print(f"[❌] Wetness Index failed: {e}")

def sanitize_dem(input_path, output_path):
    with rasterio.open(input_path) as src:
        profile = src.profile
        data = src.read(1)
        valid_data = data[np.isfinite(data) & (data > 0)]
        safe_min = valid_data.min() if valid_data.size else 1
        data = np.where(~np.isfinite(data) | (data <= 0), safe_min, data)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)
    print(f"[🛠️] Sanitized DEM saved to: {output_path}")

def try_run_wetness_index_fixed(dem_path, sca_path, output_path):
    wbt = WhiteboxTools()
    wbt.work_dir = os.path.dirname(output_path)
    wbt.verbose = True
    print("[🌊] Re-running Wetness Index with sanitized inputs...")
    try:
        wbt.run_tool("WetnessIndex", [
            f"--dem={os.path.abspath(dem_path)}",
            f"--sca={os.path.abspath(sca_path)}",
            f"--output={os.path.abspath(output_path)}"
        ])
        print(f"[✅] Wetness Index written to: {output_path}")
    except Exception as e:
        print(f"[❌] Wetness Index failed: {e}")

def compute_twi(slope_path, sca_path, output_path):
    with rasterio.open(slope_path) as slope_src, rasterio.open(sca_path) as sca_src:
        slope = slope_src.read(1)
        sca = sca_src.read(1)
        profile = slope_src.profile
    slope_rad = np.radians(slope)
    slope_rad = np.where(slope_rad <= 0, 0.001, slope_rad)
    sca = np.where(sca <= 0, 0.001, sca)
    tan_slope = np.tan(slope_rad)
    twi = np.log(sca / tan_slope)
    twi = np.where(np.isfinite(twi), twi, 0)
    profile.update(dtype=rasterio.float32)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(twi.astype(np.float32), 1)
    print(f"[✅] TWI raster saved to: {output_path}")

def regenerate_flow_accumulation_cleaned():
    import os
    dem_path = os.path.abspath("output/pune_dem_breached_clean.tif")
    out_path = os.path.abspath("output/pune_flow_accum_clean.tif")
    print("[🔁] Recomputing Flow Accumulation from breached-clean DEM...")
    wbt = WhiteboxTools()
    wbt.work_dir = os.path.dirname(out_path)
    wbt.verbose = True
    wbt.run_tool(
        "D8FlowAccumulation",
        [f"--input={dem_path}", f"--output={out_path}", "--out_type=cells"]
    )
    print(f"[✅] New Flow Accumulation saved: {out_path}")

def calculate_spi_manual(slope_path, flow_accum_path, output_path, city_name=None):
    print("[INFO] Computing SPI manually using slope and flow accumulation...")
    import rasterio
    import numpy as np
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, Resampling as WarpResampling
    import os
    if city_name:
        city_safe = city_name.replace(' ', '_')
        output_path = f"output/stream_power_index_manual_{city_safe}.tif"
    with rasterio.open(slope_path) as slope_src, rasterio.open(flow_accum_path) as acc_src:
        slope = slope_src.read(1, resampling=Resampling.bilinear)
        flow_acc = acc_src.read(1)
        # Resample flow_acc to match slope if needed
        if (slope.shape != flow_acc.shape or
            slope_src.transform != acc_src.transform or
            slope_src.crs != acc_src.crs):
            print("[INFO] Resampling flow accumulation to match slope raster...")
            flow_acc_resampled = np.empty_like(slope, dtype=np.float32)
            reproject(
                source=flow_acc,
                destination=flow_acc_resampled,
                src_transform=acc_src.transform,
                src_crs=acc_src.crs,
                dst_transform=slope_src.transform,
                dst_crs=slope_src.crs,
                resampling=WarpResampling.bilinear
            )
            flow_acc = flow_acc_resampled
        slope_rad = np.deg2rad(slope)
        spi = flow_acc * np.tan(slope_rad)
        spi = np.where(np.isfinite(spi), spi, 0)
        profile = slope_src.profile
        profile.update(dtype=rasterio.float32)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(spi.astype(np.float32), 1)
        print(f"[✅] SPI raster saved to: {output_path}")
    return output_path

def winsorize(arr, lower=1, upper=99):
    import numpy as np
    arr = np.array(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    low = np.nanpercentile(arr[finite], lower)
    high = np.nanpercentile(arr[finite], upper)
    arr = np.clip(arr, low, high)
    return arr

def normalize(array, nodata=None, winsorize_clip=True):
    import numpy as np
    arr = np.array(array, dtype=np.float32)
    # Mask nodata values
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr)
    if winsorize_clip:
        arr = winsorize(arr)
    arr_min = np.nanmin(arr[valid])
    arr_max = np.nanmax(arr[valid])
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min + 1e-9)

def compute_flood_risk_map(
    twi_path, spi_path, slope_path, pop_path, output_path, city_name=None
):
    print("[INFO] Computing Flood Risk Index with resampling, winsorizing, and improved nodata handling...")
    import rasterio
    import numpy as np
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, Resampling as WarpResampling
    import os
    import matplotlib.pyplot as plt
    import sys
    is_streamlit = 'streamlit' in sys.modules
    if is_streamlit:
        import streamlit as st
    def show_fig(fig):
        if is_streamlit:
            st.pyplot(fig)
        else:
            plt.show()
    if city_name:
        city_safe = city_name.replace(' ', '_')
        output_path = f"output/flood_risk_index_{city_safe}.tif"
    def read_and_match(base_src, target_path, desc):
        with rasterio.open(target_path) as src:
            data = src.read(1)
            nodata = src.nodata
            matched = np.empty(base_src.shape, dtype=np.float32)
            reproject(
                source=data,
                destination=matched,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=base_src.transform,
                dst_crs=base_src.crs,
                resampling=WarpResampling.bilinear
            )
            print(f"[📐] Resampled {desc} to match base shape: {matched.shape}")
            print(f"    {desc} min/max: {np.nanmin(matched)}, {np.nanmax(matched)}")
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(matched, cmap='viridis')
            ax.set_title(f"{desc} (resampled)")
            plt.colorbar(im, ax=ax)
            show_fig(fig)
            plt.close(fig)
            return matched, nodata
    with rasterio.open(slope_path) as base:
        slope = base.read(1, resampling=Resampling.bilinear)
        print("Slope min/max:", np.nanmin(slope), np.nanmax(slope))
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(slope, cmap='viridis')
        ax.set_title("Slope (resampled)")
        plt.colorbar(im, ax=ax)
        show_fig(fig)
        plt.close(fig)
        twi, twi_nodata = read_and_match(base, twi_path, "TWI")
        spi, spi_nodata = read_and_match(base, spi_path, "SPI")
        pop, pop_nodata = read_and_match(base, pop_path, "Population")
        # Winsorize and normalize each input
        twi_norm = normalize(twi, nodata=twi_nodata, winsorize_clip=True)
        spi_norm = normalize(spi, nodata=spi_nodata, winsorize_clip=True)
        slope_norm = normalize(slope, winsorize_clip=True)
        # For population, mask both nodata and extreme negative values (e.g., -99999)
        pop = np.where((pop == pop_nodata) | (pop < 0), np.nan, pop)
        pop_norm = normalize(pop, winsorize_clip=True)
        # Visualize histograms for all normalized layers and FRI
        for arr, name in zip([twi_norm, spi_norm, slope_norm, pop_norm], ["TWI norm", "SPI norm", "Slope norm", "Pop norm"]):
            print(f"{name}: min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}, std={np.nanstd(arr)}")
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(arr[np.isfinite(arr)].flatten(), bins=50, color='dodgerblue', alpha=0.7)
            ax.set_title(f"Histogram of {name}")
            ax.set_xlabel("Normalized Value")
            ax.set_ylabel("Count")
            fig.tight_layout()
            show_fig(fig)
            plt.close(fig)
        w_twi, w_spi, w_slope, w_pop = 0.3, 0.3, 0.2, 0.2
        fri = (
            w_twi * twi_norm +
            w_spi * spi_norm +
            w_slope * slope_norm +
            w_pop * pop_norm
        )
        # Mask FRI to 0 where population is 0 or nan
        fri = np.where(np.isnan(pop) | (pop == 0), 0, fri)
        print("FRI min/max:", np.nanmin(fri), np.nanmax(fri))
        # Visualize FRI histogram
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(fri[np.isfinite(fri)].flatten(), bins=50, color='crimson', alpha=0.7)
        ax.set_title("Histogram of FRI")
        ax.set_xlabel("FRI Value")
        ax.set_ylabel("Count")
        fig.tight_layout()
        show_fig(fig)
        plt.close(fig)
        # Plot normalized layers and FRI with colorbar vmin=0, vmax=1
        for arr, name, cmap in zip([twi_norm, spi_norm, slope_norm, pop_norm, fri], ["TWI norm", "SPI norm", "Slope norm", "Pop norm", "FRI"], ['Blues', 'YlGnBu', 'viridis', 'Oranges', 'Reds']):
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(name)
            plt.colorbar(im, ax=ax)
            show_fig(fig)
            plt.close(fig)
        profile = base.profile
        profile.update(dtype=rasterio.float32)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(fri.astype(np.float32), 1)
        print(f"[✅] Flood Risk Index saved to: {output_path}")
    return output_path

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile

def resample_raster(source_data, source_profile, target_profile):
    dst_array = np.empty((target_profile['height'], target_profile['width']), dtype=np.float32)
    reproject(
        source=source_data,
        destination=dst_array,
        src_transform=source_profile['transform'],
        src_crs=source_profile['crs'],
        dst_transform=target_profile['transform'],
        dst_crs=target_profile['crs'],
        resampling=Resampling.bilinear
    )
    return dst_array

def clip_lulc_to_extent(lulc_gdf, raster_profile):
    bounds = rasterio.transform.array_bounds(
        raster_profile['height'],
        raster_profile['width'],
        raster_profile['transform']
    )
    raster_box = box(*bounds)
    return lulc_gdf[lulc_gdf.intersects(raster_box)]

def plot_combined_map(fri_path, pop_path, lulc_path, city_name=None):
    import rasterio
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    from rasterio.plot import show
    import sys
    import io
    is_streamlit = 'streamlit' in sys.modules
    if is_streamlit:
        import streamlit as st
    def show_fig(fig):
        if is_streamlit:
            st.pyplot(fig)
        else:
            plt.show()
    if city_name:
        city_safe = city_name.replace(' ', '_')
        fri_path = f"output/flood_risk_index_{city_safe}.tif"
        pop_path = f"output/population_clipped_{city_safe}.tif"
        lulc_path = f"output/lulc_{city_safe}.geojson"
    fri, fri_profile = load_raster(fri_path)
    fri = np.nan_to_num(fri)
    print(f"FRI min/max: {np.nanmin(fri)}, {np.nanmax(fri)} | finite count: {np.isfinite(fri).sum()}")
    pop, pop_profile = load_raster(pop_path)
    pop_resampled = resample_raster(pop, pop_profile, fri_profile)
    # Mask nodata and negative pop before normalization
    pop_nodata = -99999.0
    pop_resampled = np.where((pop_resampled == pop_nodata) | (pop_resampled < 0), np.nan, pop_resampled)
    fri_norm = fri / np.nanmax(fri) if np.nanmax(fri) != 0 else fri
    pop_norm = pop_resampled / np.nanmax(pop_resampled[np.isfinite(pop_resampled)]) if np.nanmax(pop_resampled[np.isfinite(pop_resampled)]) != 0 else pop_resampled
    print(f"FRI norm min/max: {np.nanmin(fri_norm)}, {np.nanmax(fri_norm)} | finite count: {np.isfinite(fri_norm).sum()}")
    print(f"Pop norm min/max: {np.nanmin(pop_norm)}, {np.nanmax(pop_norm)} | finite count: {np.isfinite(pop_norm).sum()}")
    combined_score = (0.6 * fri_norm) + (0.4 * pop_norm)
    combined_score = np.clip(combined_score, 0, 1)
    print(f"Combined score min/max: {np.nanmin(combined_score)}, {np.nanmax(combined_score)} | finite count: {np.isfinite(combined_score).sum()}")
    if not np.isfinite(combined_score).any():
        print("[❌] Combined score is all NaN or invalid for this city.")
        if is_streamlit:
            st.warning("Combined flood risk map could not be generated: data is empty or invalid for this city.")
        return
    lulc = gpd.read_file(lulc_path)
    lulc_clipped = clip_lulc_to_extent(lulc, fri_profile)
    if lulc_clipped.empty:
        print("[❌] LULC is empty after clipping to raster extent.")
        if is_streamlit:
            st.warning("LULC data is empty after clipping to raster extent for this city.")
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.Reds
    show(combined_score, ax=ax, transform=fri_profile['transform'], cmap=cmap)
    if not lulc_clipped.empty:
        lulc_clipped.boundary.plot(ax=ax, edgecolor='blue', linewidth=0.8)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    cbar.set_label("Flood Risk + Population Exposure")
    plt.title("Flood Risk Map with Population and LULC Overlay")
    plt.axis("off")
    if is_streamlit:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, caption="Flood Risk Map with Population and LULC Overlay", use_column_width=True)
    else:
        show_fig(fig)
    plt.close(fig)

