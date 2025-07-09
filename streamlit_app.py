import os
import streamlit as st
from automation1 import (
    get_city_visualizations, get_and_save_lulc, plot_lulc, fetch_and_plot_hydrology,
    get_city_bbox, download_worldpop_raster, clip_and_plot_population, get_building_footprints,
    plot_buildings, calculate_flow_accumulation, fill_depressions, calculate_flow_accumulation_filled,
    fix_and_generate_wetness_index, fix_sca_raster, sanitize_dem, try_run_wetness_index_fixed,
    compute_twi, plot_combined_map, calculate_spi_manual, compute_flood_risk_map
)
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import contextily as ctx
import traceback

# === Streamlit UI Setup ===
st.set_page_config(
    page_title="Flood Risk Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        text-align: center;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        text-align: center;
        opacity: 0.9;
    }
    .map-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .map-title {
        color: #2c3e50;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .city-selection {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #218838, #1aa085);
    }
    .analysis-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    .single-map {
        grid-column: 1 / -1;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stExpander {
        background: white;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🌊 Flood Risk Assessment Tool</h1>
        <p>Comprehensive flood risk analysis for Indian cities using geospatial data and advanced modeling</p>
    </div>
""", unsafe_allow_html=True)

# === City Selection Section ===
st.markdown('<div class="city-selection">', unsafe_allow_html=True)
st.markdown("### 🏙️ Select City for Analysis")

# Define city categories
full_analysis_cities = [
    "Pune", "Mumbai", "Thane", "Delhi", "Bangalore", 
    "Hyderabad", "Kolkata", "Kanpur", "Nagpur"
]

limited_analysis_cities = [
    "Chennai", "Surat", "Bhopal", "Jaipur", 
    "Indore", "Lucknow"
]

# City selection interface
col1, col2 = st.columns([1, 1])

with col1:
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Full Analysis", "Limited Analysis"],
        help="Full Analysis: All maps available | Limited Analysis: Selected maps only"
    )

with col2:
    if analysis_type == "Full Analysis":
        selected_city = st.selectbox(
            "Select city:",
            full_analysis_cities,
            index=0
        )
        st.markdown('<div class="info-card">✅ <strong>Full analysis includes:</strong> All 10 maps with comprehensive risk assessment</div>', unsafe_allow_html=True)
    else:
        selected_city = st.selectbox(
            "Select city:",
            limited_analysis_cities,
            index=0
        )
        st.markdown('<div class="warning-card">⚠️ <strong>Limited analysis includes:</strong> Basic maps only (DEM, LULC, Hydrology, Flow Accumulation)</div>', unsafe_allow_html=True)

# Data availability notice
with st.expander("⚠️ Data Availability & Limitations"):
    st.markdown("""
    **Please note:** Some maps may not be available or accurate for certain cities due to:
    - **Data unavailability** from open sources
    - **Large datasets** requiring extensive processing time
    - **Computing limitations** in real-time processing
    - **Geographic constraints** for specific regions
    """)

# Generate button
generate_analysis = st.button("🔍 Generate Flood Risk Analysis", type="primary")

st.markdown('</div>', unsafe_allow_html=True)



# === Analysis Results ===
if generate_analysis and selected_city:
    city_name = selected_city
    city_safe = city_name.replace(' ', '_')
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.markdown(f"## 📊 Flood Risk Analysis Results for {city_name}")
    
    # === Basic Maps (Available for both analysis types) ===
    st.markdown("### 🗺️ Basic Terrain & Land Use Analysis")
    
    # DEM & LULC in grid layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="map-section">', unsafe_allow_html=True)
        st.markdown('<div class="map-title">🗻 DEM & Terrain</div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About DEM & Terrain"):
            st.markdown("""
            **Purpose:** Shows elevation, slope, and terrain characteristics
            **Calculation:** SRTM elevation data with terrain derivatives
            **Importance:** Identifies topography and water flow patterns
            """)
        
        status_text.text("Generating DEM & Terrain...")
        progress_bar.progress(10)
        
        try:
            figs = get_city_visualizations(city_name)
            for fig in figs:
                fig.set_size_inches(8, 6)
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate DEM visualizations: {str(e)[:50]}...")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="map-section">', unsafe_allow_html=True)
        st.markdown('<div class="map-title">🗺️ Land Use / Land Cover</div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About LULC"):
            st.markdown("""
            **Purpose:** Land use classification (urban, water, vegetation)
            **Source:** OpenStreetMap land use data
            **Importance:** Shows built-up areas and natural features
            """)
        
        status_text.text("Generating LULC map...")
        progress_bar.progress(20)
        
        try:
            lulc_gdf = get_and_save_lulc(city_name)
            if lulc_gdf is not None and not lulc_gdf.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                lulc_gdf.to_crs(epsg=3857).plot(
                    column="lulc_category", cmap="Set3", legend=True, ax=ax,
                    edgecolor="k", alpha=0.7
                )
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                ax.set_title(f"LULC Map - {city_name}")
                plt.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No LULC data available")
        except Exception as e:
            st.warning(f"Could not generate LULC map: {str(e)[:50]}...")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Hydrology & Flow Accumulation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="map-section">', unsafe_allow_html=True)
        st.markdown('<div class="map-title">💧 Hydrological Features</div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About Hydrology"):
            st.markdown("""
            **Purpose:** Rivers, streams, lakes, and water bodies
            **Source:** OpenStreetMap water features
            **Importance:** Natural drainage systems for flood assessment
            """)
        
        status_text.text("Generating hydrological features...")
        progress_bar.progress(30)
        
        try:
            hydro_results = fetch_and_plot_hydrology(city_name)
            hydro_map_path = hydro_results.get("figure")
            if hydro_map_path and os.path.exists(hydro_map_path):
                st.image(hydro_map_path, use_container_width=True)
            else:
                st.info("No hydrological features available")
        except Exception as e:
            st.warning(f"Could not generate hydrological map: {str(e)[:50]}...")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="map-section">', unsafe_allow_html=True)
        st.markdown('<div class="map-title">🌊 Flow Accumulation</div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About Flow Accumulation"):
            st.markdown("""
            **Purpose:** Water accumulation patterns from DEM
            **Calculation:** Flow direction algorithms
            **Importance:** Identifies natural drainage paths
            """)
        
        status_text.text("Generating flow accumulation...")
        progress_bar.progress(40)
        
        dem_path = os.path.join("output", "dem.tif")
        flow_accum_path = os.path.join("output", "flow_accumulation.tif")
        
        try:
            if os.path.exists(dem_path) and not os.path.exists(flow_accum_path):
                calculate_flow_accumulation(dem_path, output_dir="output")
            
            if os.path.exists(flow_accum_path):
                with rasterio.open(flow_accum_path) as src:
                    arr = src.read(1)
                    arr = np.where(np.isfinite(arr), arr, np.nan)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(arr, cmap="Blues", 
                                 vmin=np.nanpercentile(arr, 2), 
                                 vmax=np.nanpercentile(arr, 98))
                    plt.colorbar(im, ax=ax, label="Flow Accumulation")
                    ax.set_title("Flow Accumulation")
                    ax.set_axis_off()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                st.info("Flow accumulation data not available")
        except Exception as e:
            st.warning(f"Could not generate flow accumulation: {str(e)[:50]}...")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # === Advanced Analysis (Full Analysis Only) ===
    if analysis_type == "Full Analysis":
        st.markdown("### 📈 Advanced Risk Assessment")
        
        # Population & Buildings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="map-section">', unsafe_allow_html=True)
            st.markdown('<div class="map-title">👥 Population Density</div>', unsafe_allow_html=True)
            
            status_text.text("Generating population density...")
            progress_bar.progress(50)
            
            try:
                bbox, city_gdf = get_city_bbox(city_name)
                tif_path = download_worldpop_raster()
                clip_and_plot_population(city_gdf, tif_path, city_name=city_name)
                pop_plot_path = os.path.join("output", f"{city_safe}_population.png")
                if os.path.exists(pop_plot_path):
                    st.image(pop_plot_path, use_container_width=True)
                else:
                    st.info("Population data not available")
            except Exception as e:
                st.warning(f"Could not generate population map: {str(e)[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="map-section">', unsafe_allow_html=True)
            st.markdown('<div class="map-title">🏢 Building Footprints</div>', unsafe_allow_html=True)
            
            status_text.text("Generating building footprints...")
            progress_bar.progress(60)
            
            try:
                buildings_gdf = get_building_footprints(city_name)
                plot_buildings(buildings_gdf, title=f"Building Footprints: {city_name}")
                building_plot_path = os.path.join("output", f"Building_Footprints_{city_safe}.png")
                if os.path.exists(building_plot_path):
                    st.image(building_plot_path, use_container_width=True)
                else:
                    st.info("Building footprints not available")
            except Exception as e:
                st.warning(f"Could not generate building footprints: {str(e)[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk Indices
        st.markdown("### 🎯 Risk Indices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="map-section">', unsafe_allow_html=True)
            st.markdown('<div class="map-title">💧 Topographic Wetness Index</div>', unsafe_allow_html=True)
            
            status_text.text("Generating TWI...")
            progress_bar.progress(70)
            
            dem_path = os.path.join("output", "dem.tif")
            wetness_index_path = os.path.join("output", "wetness_index.tif")
            
            try:
                if not os.path.exists(wetness_index_path):
                    fix_and_generate_wetness_index(dem_path, output_dir="output")
                
                if os.path.exists(wetness_index_path):
                    with rasterio.open(wetness_index_path) as src:
                        arr = src.read(1)
                        arr = np.where(np.isfinite(arr), arr, np.nan)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(arr, cmap="Blues", 
                                     vmin=np.nanpercentile(arr, 2), 
                                     vmax=np.nanpercentile(arr, 98))
                        plt.colorbar(im, ax=ax, label="TWI Value")
                        ax.set_title("Topographic Wetness Index")
                        ax.set_axis_off()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.info("TWI data not available")
            except Exception as e:
                st.warning(f"Could not generate TWI: {str(e)[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="map-section">', unsafe_allow_html=True)
            st.markdown('<div class="map-title">⚡ Stream Power Index</div>', unsafe_allow_html=True)
            
            status_text.text("Generating SPI...")
            progress_bar.progress(80)
            
            spi_path = os.path.join("output", f"stream_power_index_manual_{city_safe}.tif")
            slope_path = os.path.join("output", "slope.tif")
            fixed_sca_path = os.path.join("output", "flow_accum_breached_fixed.tif")
            
            try:
                if not os.path.exists(spi_path) and os.path.exists(slope_path) and os.path.exists(fixed_sca_path):
                    calculate_spi_manual(slope_path, fixed_sca_path, spi_path, city_name=city_name)
                
                if os.path.exists(spi_path):
                    with rasterio.open(spi_path) as src:
                        arr = src.read(1)
                        arr = np.where(np.isfinite(arr), arr, np.nan)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(arr, cmap="YlGnBu", 
                                     vmin=np.nanpercentile(arr, 2), 
                                     vmax=np.nanpercentile(arr, 98))
                        plt.colorbar(im, ax=ax, label="SPI Value")
                        ax.set_title("Stream Power Index")
                        ax.set_axis_off()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.info("SPI data not available")
            except Exception as e:
                st.warning(f"Could not generate SPI: {str(e)[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Final Risk Assessment
        st.markdown("### 🚨 Final Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="map-section">', unsafe_allow_html=True)
            st.markdown('<div class="map-title">🚨 Flood Risk Index</div>', unsafe_allow_html=True)
            
            status_text.text("Generating Flood Risk Index...")
            progress_bar.progress(90)
            
            fri_path = os.path.join("output", f"flood_risk_index_{city_safe}.tif")
            pop_path = os.path.join("output", f"population_clipped_{city_safe}.tif")
            
            try:
                if not os.path.exists(fri_path) and all(os.path.exists(p) for p in [wetness_index_path, spi_path, slope_path, pop_path]):
                    compute_flood_risk_map(
                        twi_path=wetness_index_path,
                        spi_path=spi_path,
                        slope_path=slope_path,
                        pop_path=pop_path,
                        output_path=fri_path,
                        city_name=city_name
                    )
                
                if os.path.exists(fri_path):
                    with rasterio.open(fri_path) as src:
                        arr = src.read(1)
                        arr = np.where(np.isfinite(arr), arr, np.nan)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(arr, cmap="OrRd", 
                                     vmin=np.nanpercentile(arr, 2), 
                                     vmax=np.nanpercentile(arr, 98))
                        plt.colorbar(im, ax=ax, label="Risk Level")
                        ax.set_title("Flood Risk Index")
                        ax.set_axis_off()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.info("FRI data not available")
            except Exception as e:
                st.warning(f"Could not generate FRI: {str(e)[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="map-section">', unsafe_allow_html=True)
            st.markdown('<div class="map-title">🌊 Combined Risk Assessment</div>', unsafe_allow_html=True)
            
            status_text.text("Generating combined assessment...")
            progress_bar.progress(95)
            
            lulc_path = os.path.join("output", f"lulc_{city_safe}.geojson")
            try:
                if os.path.exists(fri_path) and os.path.exists(pop_path) and os.path.exists(lulc_path):
                    plot_combined_map(fri_path, pop_path, lulc_path, city_name=city_name)
                    combined_plot_path = os.path.join("output", f"combined_flood_risk_{city_safe}.png")
                    if os.path.exists(combined_plot_path):
                        st.image(combined_plot_path, use_container_width=True)
                    else:
                        st.info("Combined map not available")
                else:
                    st.info("Combined analysis requires all component maps")
            except Exception as e:
                st.warning(f"Could not generate combined map: {str(e)[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Future validation notice
    st.markdown("---")
    st.markdown('<div class="warning-card">', unsafe_allow_html=True)
    st.markdown("""
    **🔬 Future Validation & Enhancement:**
    
    We are working on improving accuracy through:
    - **Historical Data Integration:** Incorporating flood records and news data
    - **Vision Model Analysis:** Using AI to compare predictions with actual flood imagery
    - **Real-time Validation:** Cross-referencing with documented incidents
    """)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("👆 **Ready to start?** Select a city above and click 'Generate Flood Risk Analysis' to begin the comprehensive assessment.")
    st.markdown('</div>', unsafe_allow_html=True)